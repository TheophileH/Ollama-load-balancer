mod discovery;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{any},
    Router,
};
use reqwest::Client;
use serde_json::Value;
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use dotenv::dotenv;
use discovery::{BackendRegistry, BackendType, start_discovery};
use rand::seq::SliceRandom;
use futures_util::future::join_all;
use std::pin::Pin;
use std::task::{Context, Poll};
use futures_util::Stream;
use std::cmp::Reverse;

// Shared state now holds the dynamic registry
#[derive(Clone)]
struct AppState {
    client: Client,
    registry: BackendRegistry,
    default_strategy: String,
    enable_affinity: bool,
    nvidia_vram_limit: u64,
    amd_vram_limit: u64,
}

#[tokio::main]
async fn main() {
    dotenv().ok();
    
    // Initialize tracing
    // prioritized: LOG_LEVEL > RUST_LOG > "info"
    let log_var = env::var("LOG_LEVEL").or_else(|_| env::var("RUST_LOG")).unwrap_or_else(|_| "info".to_string());
    let filter = tracing_subscriber::EnvFilter::new(log_var);

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    let port = env::var("PORT").unwrap_or_else(|_| "11434".to_string());
    let default_strategy = env::var("LOAD_BALANCING_STRATEGY").unwrap_or_else(|_| "random".to_string());
    let enable_affinity = env::var("OLLAMA_ENABLE_AFFINITY").unwrap_or_else(|_| "false".to_string()) == "true";
    
    // Parse VRAM limits (MB -> Bytes)
    let nvidia_vram_mb: u64 = env::var("OLLAMA_NVIDIA_VRAM_MB").unwrap_or("0".to_string()).parse().unwrap_or(0);
    let amd_vram_mb: u64 = env::var("OLLAMA_AMD_VRAM_MB").unwrap_or("0".to_string()).parse().unwrap_or(0);
    
    let nvidia_vram_limit = nvidia_vram_mb * 1024 * 1024;
    let amd_vram_limit = amd_vram_mb * 1024 * 1024;

    let client = Client::builder()
        .build()
        .expect("Failed to create HTTP client");

    // Initialize Registry and Start Discovery
    let registry = Arc::new(RwLock::new(Vec::new()));
    start_discovery(registry.clone()).await;

    let state = Arc::new(AppState {
        client,
        registry,
        default_strategy,
        enable_affinity,
        nvidia_vram_limit,
        amd_vram_limit,
    });

    let app = Router::new()
        // Catch-all route for everything
        .route("/*path", any(handler))
        .route("/", any(handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handler(
    State(state): State<Arc<AppState>>,
    req: Request,
) -> Result<Response, StatusCode> {
    // 1. Determine Intent (Preferred Vendor & Model Name)
    let (req, preferred_type, model_name) = determine_intent(req).await;
    
    let path = req.uri().path().to_string();
    let method = req.method().clone();
    
    // 2. Select Specific Backend Instance
    let reg = state.registry.read().await;
    
    let mut candidates: Vec<&discovery::Backend> = reg.iter()
        .filter(|b| b.backend_type == preferred_type)
        .collect();

    // If no candidates for preferred, fallback to all if not enforced (TODO: enforce check)
    if candidates.is_empty() {
         tracing::warn!("Preferred backend {:?} not found! Falling back to any.", preferred_type);
         candidates = reg.iter().collect();
    }

    if candidates.is_empty() {
        tracing::error!("No backends available!");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    // Load Balancing Logic
    let mut selected_backend = None;

    if state.enable_affinity && model_name.is_some() {
        let model = model_name.as_ref().unwrap();
        
        // Parallel check for loaded models
        let mut checks = Vec::new();
        for c in &candidates {
            checks.push(check_if_model_loaded(&state.client, &c.ip, c.port, model));
        }
        
        let results = join_all(checks).await;
        
        // Zip and Score: (Has Model, -Active Requests)
        let mut scored: Vec<(bool, usize, &discovery::Backend)> = Vec::new();
        
        for (i, is_loaded) in results.into_iter().enumerate() {
            let loaded = is_loaded.unwrap_or(false);
            let active = candidates[i].active_requests.load(Ordering::Relaxed);
            scored.push((loaded, active, candidates[i]));
        }
        
        // Sort: We want Loaded=true first. Then Active=Low first.
        scored.sort_by_key(|(loaded, active, _)| (*loaded, Reverse(*active)));
        
        // Take best ONLY if it is loaded
        if let Some((true, _, b)) = scored.last() {
             tracing::info!("Affinity hit! Routing to {} for model {}", b.id, model);
             selected_backend = Some((*b).clone());
        }
    }

    if selected_backend.is_none() {
        // Fallback Strategy
        if state.default_strategy == "vram" {
             // Parallel Check VRAM
             let mut checks = Vec::new();
             for c in &candidates {
                 checks.push(get_used_vram(&state.client, &c.ip, c.port));
             }
             
             let results = join_all(checks).await;
             
             // (Free VRAM, &Backend)
             let mut vram_stats: Vec<(u64, &discovery::Backend)> = Vec::new();
             
             for (i, used_opt) in results.into_iter().enumerate() {
                 if let Some(used) = used_opt {
                     let backend = candidates[i];
                     
                     // 1. Try Autodetected Capacity
                     // 2. Fallback to Global Env Config
                     let limit = if backend.vram_capacity > 0 {
                         backend.vram_capacity
                     } else {
                         match backend.backend_type {
                             BackendType::Nvidia => state.nvidia_vram_limit,
                             BackendType::Amd => state.amd_vram_limit,
                         }
                     };
                     
                     if limit > used {
                         vram_stats.push((limit - used, backend));
                     } else {
                         vram_stats.push((0, backend));
                     }
                 }
             }
             
             // Sort by Free VRAM Descending
             vram_stats.sort_by_key(|(free, _)| Reverse(*free));
             
             if let Some((free, b)) = vram_stats.first() {
                  tracing::info!("VRAM Strategy: Selected {} with {} MB free", b.id, free / (1024*1024));
                  selected_backend = Some((*b).clone());
             }
        }
        
        // If still none (or strategy != vram), use Random
        if selected_backend.is_none() {
            selected_backend = candidates.choose(&mut rand::thread_rng()).cloned().map(|b| b.clone());
        }
    }

    let target_backend = match selected_backend {
        Some(b) => b,
        None => return Err(StatusCode::SERVICE_UNAVAILABLE),
    };

    let target_url = format!("http://{}:{}{}", target_backend.ip, target_backend.port, path);
    tracing::info!("Proxying {} {} -> {} (Active: {})", method, path, target_url, target_backend.active_requests.load(Ordering::Relaxed));

    // Increment Active Requests
    target_backend.active_requests.fetch_add(1, Ordering::Relaxed);
    let guard = RequestGuard::new(target_backend.active_requests.clone());

    // Proxy the request
    let headers = req.headers().clone();
    let body = req.into_body();

    let client_req = state.client
        .request(method, &target_url)
        .body(reqwest::Body::wrap_stream(body.into_data_stream()));
        
    let mut final_req = client_req;
    for (key, value) in headers.iter() {
        if key != "host" && key != "content-length" {
             final_req = final_req.header(key, value);
        }
    }

    match final_req.send().await {
        Ok(res) => {
            let status = res.status();
            let headers = res.headers().clone();
            let stream = res.bytes_stream();
            
            // Wrap stream to hold guard
            let measured_stream = MeasuredStream {
                stream,
                _guard: guard,
            };
            
            let body = Body::from_stream(measured_stream);
            
            let mut response = Response::builder()
                .status(status)
                .body(body)
                .unwrap();
                
            *response.headers_mut() = headers;
            Ok(response)
        },
        Err(e) => {
            tracing::error!("Proxy error: {}", e);
            // Request failed immediately, guard dropped here, count decremented. Correct.
            Err(StatusCode::BAD_GATEWAY)
        }
    }
}

// Check if model is loaded on a node
async fn check_if_model_loaded(client: &Client, ip: &str, port: u16, model: &str) -> Option<bool> {
    // We assume /api/ps returns running models
    // Fast timeout is essential here
    let url = format!("http://{}:{}/api/ps", ip, port);
    
    match client.get(&url).timeout(std::time::Duration::from_millis(500)).send().await {
        Ok(res) => {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(models) = json["models"].as_array() {
                     for m in models {
                         if let Some(name) = m["name"].as_str() {
                             // Match exact or prefix
                             if name == model || name.starts_with(model) {
                                 return Some(true);
                             }
                         }
                     }
                }
            }
            Some(false)
        },
        Err(_) => None, // Failed to check
    }
}

async fn get_used_vram(client: &Client, ip: &str, port: u16) -> Option<u64> {
    let url = format!("http://{}:{}/api/ps", ip, port);
    match client.get(&url).timeout(std::time::Duration::from_millis(500)).send().await {
         Ok(res) => {
             if let Ok(json) = res.json::<Value>().await {
                 let mut used = 0;
                 if let Some(models) = json["models"].as_array() {
                      for m in models {
                           // Try size_vram first, then size, then 0
                           let size = m.get("size_vram").and_then(|v| v.as_u64())
                                      .or_else(|| m.get("size").and_then(|v| v.as_u64()))
                                      .unwrap_or(0);
                           used += size;
                      }
                 }
                 return Some(used);
             }
             None
         },
         Err(_) => None,
    }
}

// Guard to decrement active requests on Drop
struct RequestGuard {
    counter: Arc<AtomicUsize>,
}

impl RequestGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        Self { counter }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

// MeasuredStream wrapper
struct MeasuredStream<S> {
    stream: S,
    _guard: RequestGuard, // Held until stream is dropped
}

impl<S, B, E> Stream for MeasuredStream<S>
where
    S: Stream<Item = Result<B, E>> + Unpin,
{
    type Item = Result<B, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

// Returns the request (possibly modified), preferred backend type, AND model name found
async fn determine_intent(req: Request) -> (Request, BackendType, Option<String>) {
    let (mut parts, body) = req.into_parts();
    let mut intent = BackendType::Nvidia; 
    let mut found_model = None;
    
    // Only parse body for API methods that send JSON
    let path = parts.uri.path();
    if !path.starts_with("/api/generate") && !path.starts_with("/api/chat") && !path.starts_with("/api/show") && !path.starts_with("/api/pull") {
         let req = Request::from_parts(parts, body);
         return (req, intent, None);
    }

    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(_) => return (Request::from_parts(parts, Body::empty()), intent, None),
    };
    
    let mut modified_bytes = bytes.clone();

    if let Ok(mut json) = serde_json::from_slice::<Value>(&bytes) {
        // Determine field name (prioritize "model" if non-empty, else "name")
        let field_name = if json.get("model").and_then(|v| v.as_str()).map(|s| !s.is_empty()).unwrap_or(false) {
            Some("model")
        } else if json.get("name").is_some() {
            Some("name")
        } else {
            None
        };

        if let Some(field) = field_name {
            if let Some(model_str) = json[field].as_str() {
                let model = model_str.to_string();
                let mut new_model = model.clone();
                let mut matched = false;
                
                if let Some(stripped) = model.strip_prefix("nvidia/") {
                    intent = BackendType::Nvidia;
                    new_model = stripped.to_string();
                    matched = true;
                } else if let Some(stripped) = model.strip_prefix("cuda/") {
                    intent = BackendType::Nvidia;
                    new_model = stripped.to_string();
                    matched = true;
                } else if let Some(stripped) = model.strip_prefix("amd/") {
                    intent = BackendType::Amd;
                    new_model = stripped.to_string();
                    matched = true;
                } else if let Some(stripped) = model.strip_prefix("rocm/") {
                     intent = BackendType::Amd;
                     new_model = stripped.to_string();
                     matched = true;
                } else if let Some((base, _)) = model.rsplit_once("@nvidia") {
                    intent = BackendType::Nvidia;
                    new_model = base.to_string();
                    matched = true;
                } else if let Some((base, _)) = model.rsplit_once("@amd") {
                    intent = BackendType::Amd;
                    new_model = base.to_string();
                    matched = true;
                }
                
                // Set found model
                found_model = Some(new_model.clone());

                // If changed, update JSON and bytes
                if matched && new_model != model {
                    json[field] = Value::String(new_model);
                    // Remove Content-Length as size changes
                    parts.headers.remove("content-length");
                    if let Ok(vec) = serde_json::to_vec(&json) {
                         modified_bytes = vec.into();
                    }
                }
            }
        }
    }

    let req = Request::from_parts(parts, Body::from(modified_bytes));
    (req, intent, found_model)
}

