mod discovery;
mod vram_monitor;
mod gpu_utilization_monitor;


use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::Response,
    routing::{any},
    Router,
};
use reqwest::Client;
use serde_json::Value;
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU8, Ordering};
use std::time::Duration;
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
    model_affinity: bool,
    enforce_strategy: bool,
    nvidia_vram_limit: u64,
    amd_vram_limit: u64,
    last_backend_type: Arc<AtomicU8>, // 0 = Nvidia, 1 = Amd
    round_robin_counter: Arc<AtomicUsize>, // Counter for round-robin
    gpu_util_state: gpu_utilization_monitor::GpuUtilState,
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
    let model_affinity = env::var("OLLAMA_MODEL_AFFINITY").unwrap_or_else(|_| "false".to_string()) == "true";
    let enforce_strategy = env::var("ENFORCE_STRATEGY").unwrap_or_else(|_| "true".to_string()) == "true";
    
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

    // Create shared GPU utilization state BEFORE AppState (will be populated by monitor)
    let gpu_util_state = Arc::new(RwLock::new(Vec::new()));

    let state = Arc::new(AppState {
        client,
        registry: registry.clone(),
        default_strategy,
        model_affinity,
        enforce_strategy,
        nvidia_vram_limit,
        amd_vram_limit,
        last_backend_type: Arc::new(AtomicU8::new(0)), // Start with Nvidia
        round_robin_counter: Arc::new(AtomicUsize::new(0)), // Start at 0
        gpu_util_state: gpu_util_state.clone(),
    });

    let app = Router::new()
        // Specialized routes for aggregation
        .route("/api/ps", any(handle_ps))
        .route("/api/tags", any(handle_tags))
        .route("/api/stats", any(handle_stats))
        // Catch-all route for everything else
        .route("/*path", any(handler))
        .route("/", any(handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone()); // Clone state for app

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Listening on {}", addr);
    
    // Start discovery
    let registry_clone = state.registry.clone();
    tokio::spawn(async move {
        discovery::start_discovery(registry_clone).await;
    });
    
    // Start VRAM monitor (polls every 2 seconds for real-time updates)
    let registry_clone2 = state.registry.clone();
    let client_clone = state.client.clone();
    tokio::spawn(async move {
        vram_monitor::start_vram_monitor(client_clone, registry_clone2, 2).await;
    });
    
    // Start GPU utilization monitor (polls nvidia-smi/rocm-smi every 1 second)
    let gpu_util_clone = state.gpu_util_state.clone();
    tokio::spawn(async move {
        gpu_utilization_monitor::start_gpu_monitor(gpu_util_clone).await;
    });
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_ps(
    State(state): State<Arc<AppState>>,
) -> Result<Response, StatusCode> {
    aggregate_responses(state, "/api/ps", "models").await
}

async fn handle_tags(
    State(state): State<Arc<AppState>>,
) -> Result<Response, StatusCode> {
    aggregate_responses(state, "/api/tags", "models").await
}

async fn handle_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Response, StatusCode> {
    use serde_json::json;
    
    let reg = state.registry.read().await;
    let mut backend_stats = Vec::new();
    
    for backend in reg.iter() {
        let url = format!("http://{}:{}", backend.ip, backend.port);
        
        // Get running models from backend
        let mut models = Vec::new();
        match state.client.get(&format!("{}/api/ps", url))
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(res) => {
                if let Ok(json) = res.json::<Value>().await {
                    if let Some(model_array) = json["models"].as_array() {
                        for m in model_array {
                            if let Some(name) = m["name"].as_str() {
                                models.push(name.to_string());
                            }
                        }
                    }
                }
            }
            Err(_) => {}
        }
        
        // Get GPU stats
        let gpus = backend.gpus.read().await;
        let mut total_vram_mb = 0u64;
        let mut used_vram_mb = 0u64;
        let mut free_vram_mb = 0u64;
        let mut gpu_details = Vec::new();
        
        for gpu in gpus.iter() {
            let total = gpu.total_memory;
            let free = gpu.free_memory;
            let used = total.saturating_sub(free);
            
            total_vram_mb += total / (1024 * 1024);
            free_vram_mb += free / (1024 * 1024);
            used_vram_mb += used / (1024 * 1024);
            
            gpu_details.push(json!({
                "id": gpu.device_index,
                "total_mb": total / (1024 * 1024),
                "used_mb": used / (1024 * 1024),
                "free_mb": free / (1024 * 1024)
            }));
        }
        
        // If no GPU data yet, use fallback
        if gpus.is_empty() {
            total_vram_mb = backend.vram_capacity / (1024 * 1024);
            // Try to get usage from /api/ps
            if let Some(used) = get_used_vram(&state.client, &backend.ip, backend.port).await {
                used_vram_mb = used / (1024 * 1024);
                free_vram_mb = total_vram_mb.saturating_sub(used_vram_mb);
            } else {
                free_vram_mb = total_vram_mb;
            }
        }
        
        let backend_data = json!({
            "id": backend.id,
            "type": format!("{:?}", backend.backend_type).to_lowercase(),
            "url": url,
            "active_requests": backend.active_requests.load(Ordering::Relaxed),
            "models": models,
            "vram": {
                "total_mb": total_vram_mb,
                "used_mb": used_vram_mb,
                "free_mb": free_vram_mb,
                "gpus": gpu_details
            }
        });
        
        backend_stats.push(backend_data);
    }
    
    let response_json = json!({
        "backends": backend_stats
    });
    
    Ok(Response::builder()
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&response_json).unwrap()))
        .unwrap())
}

async fn aggregate_responses(
    state: Arc<AppState>,
    path: &str,
    key: &str,
) -> Result<Response, StatusCode> {
    let reg = state.registry.read().await;
    let mut futures = Vec::new();

    for backend in reg.iter() {
        let url = format!("http://{}:{}{}", backend.ip, backend.port, path);
        let client = state.client.clone();
        futures.push(async move {
            match client.get(&url).timeout(std::time::Duration::from_secs(2)).send().await {
                Ok(res) => res.json::<Value>().await.ok(),
                Err(_) => None,
            }
        });
    }
    
    let results = join_all(futures).await;
    
    // Use HashMap for deduplication by model name
    use std::collections::HashMap;
    let mut models_map: HashMap<String, Value> = HashMap::new();

    for res in results {
        if let Some(json) = res {
            if let Some(array) = json[key].as_array() {
                for item in array {
                    // Extract model name for deduplication
                    if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                        // Only insert if not already present (first occurrence wins)
                        models_map.entry(name.to_string()).or_insert_with(|| item.clone());
                    } else {
                        // If no name field, keep it anyway (shouldn't happen for models)
                        models_map.insert(format!("unnamed_{}", models_map.len()), item.clone());
                    }
                }
            }
        }
    }
    
    // Convert HashMap back to Vec
    let aggregated_list: Vec<Value> = models_map.into_values().collect();
    
    let mut final_json = serde_json::Map::new();
    final_json.insert(key.to_string(), Value::Array(aggregated_list));
    
    Ok(Response::builder()
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&final_json).unwrap()))
        .unwrap())
}

async fn handler(
    State(state): State<Arc<AppState>>,
    req: Request,
) -> Result<Response, StatusCode> {
    // 1. Determine Intent (Preferred Vendor & Model Name & Indices)
    let (req, preferred_type, model_name, target_indices) = determine_intent(req).await;
    
    let path = req.uri().path().to_string();
    let method = req.method().clone();
    
    // Log incoming request
    if let Some(ref model) = model_name {
        if let Some(pref) = preferred_type {
            tracing::info!("📥 Incoming {} {} | Model: '{}' | Preferred: {:?}", method, path, model, pref);
        } else {
            tracing::info!("📥 Incoming {} {} | Model: '{}' | Checking all backends", method, path, model);
        }
    } else {
        tracing::debug!("📥 Incoming {} {} | No model detected", method, path);
    }
    
    // 2. Select Specific Backend Instance
    let reg = state.registry.read().await;
    
    // Only filter by type if explicitly requested; otherwise check ALL backends for VRAM strategy
    let mut candidates: Vec<&discovery::Backend> = if let Some(pref_type) = preferred_type {
        reg.iter().filter(|b| b.backend_type == pref_type).collect()
    } else {
        reg.iter().collect()  // All backends
    };

    tracing::debug!("Found {} {:?} backends in registry", candidates.len(), preferred_type);

    // Sort candidates by NAME to ensure deterministic indexing (e.g. ollama-amd-1, ollama-amd-2)
    candidates.sort_by_key(|b| &b.name);

    // If specific indices requested (Subset Routing)
    if let Some(indices) = target_indices {
        let mut subset = Vec::new();
        for &idx in &indices {
            if let Some(backend) = candidates.get(idx) {
                subset.push(*backend);
            } else {
                tracing::warn!("Requested index {} out of bounds (available: {})", idx, candidates.len());
            }
        }
        if !subset.is_empty() {
            candidates = subset;
            tracing::info!("Routing restricted to {} specific instances requested", candidates.len());
        } else {
             tracing::error!("Target indices {:?} invalid / not found!", indices);
             return Err(StatusCode::BAD_REQUEST);
        }
    }

    // If no candidates for preferred/requested, fallback only if indices weren't enforced
    if candidates.is_empty() {
         tracing::warn!("Preferred backend {:?} not found!", preferred_type);
         // Do not fallback if specific indices failed, that's a user error usually.
         // But for general type, fallback might be desired? 
         // Let's assume strict intent if indices provided, looser if not.
         // Here we just check empty again.
    }

    if candidates.is_empty() {
        tracing::error!("No backends available!");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    // PREDICTIVE VRAM REQUIREMENTS CHECK
    // Query model size and filter backends with insufficient VRAM
    // IMPORTANT: Check ALL backends (both NVIDIA and AMD), not just preferred type!
    let mut vram_filtered_candidates;
    
    if let Some(ref model) = model_name {
        // Query model size from any available backend
        let all_backends_for_query: Vec<&discovery::Backend> = reg.iter().collect();
        
        if let Some(sample_backend) = all_backends_for_query.first() {
            if let Some(required_vram) = get_model_vram_requirement(&state.client, sample_backend, model).await {
                let required_gb = required_vram as f64 / (1024.0 * 1024.0 * 1024.0);
                
                // Filter candidates by available VRAM
                // CRITICAL: Use REAL-TIME host GPU monitoring data (NOT stale container logs)!
                vram_filtered_candidates = Vec::new();
                
                for backend in &candidates {
                    // Get REAL-TIME data from host GPU monitoring
                    // Works for BOTH NVIDIA (nvidia-smi) and AMD (rocm-smi)
                    let host_gpus = state.gpu_util_state.read().await;
                    
                    let total_free_vram: u64 = if !host_gpus.is_empty() {
                        // Host monitoring data is available - USE THIS!
                        // PRIORITY: Use visible_gpu_indices (accurate, complete)
                        // FALLBACK: Use gpus from logs (may be incomplete)
                        
                        if !backend.visible_gpu_indices.is_empty() {
                            // Best path: Sum all visible GPUs using host monitoring
                            backend.visible_gpu_indices.iter()
                                .filter_map(|&idx| {
                                    host_gpus.iter().find(|h| h.gpu_index == idx && h.backend_type == backend.backend_type)
                                })
                                .map(|host_gpu| {
                                    let free_mb = host_gpu.memory_total_mb.saturating_sub(host_gpu.memory_used_mb);
                                    free_mb * 1024 * 1024
                                })
                                .sum()
                        } else {
                            // Fallback: Use log-parsed GPU data if discovery failed
                            let gpus = backend.gpus.read().await;
                            
                            if !gpus.is_empty() {
                                // Map container GPUs to host GPUs
                                gpus.iter()
                                    .filter_map(|container_gpu| {
                                        host_gpus.iter()
                                            .find(|h| h.gpu_index == container_gpu.device_index)
                                    })
                                    .map(|host_gpu| {
                                        let free_mb = host_gpu.memory_total_mb.saturating_sub(host_gpu.memory_used_mb);
                                        free_mb * 1024 * 1024
                                    })
                                    .sum()
                            } else {
                                // Ultimate fallback: static capacity
                                backend.vram_capacity
                            }
                        }
                    } else {
                        // Host monitoring not available (both nvidia-smi and rocm-smi failed)
                        // Fall back to log-parsed data (works for NVIDIA and AMD)
                        let gpus = backend.gpus.read().await;
                        if !gpus.is_empty() {
                            gpus.iter().map(|gpu| gpu.free_memory).sum()
                        } else {
                            // Ultimate fallback: static capacity
                            backend.vram_capacity
                        }
                    };
                    
                    let free_gb = total_free_vram as f64 / (1024.0 * 1024.0 * 1024.0);
                    
                    if total_free_vram >= required_vram {
                        vram_filtered_candidates.push(*backend);
                        tracing::debug!(
                            "✅ {} has {:.2}GB total free (>= {:.2}GB required) [source: {}]",
                            backend.name,
                            free_gb,
                            required_gb,
                            if !host_gpus.is_empty() { "real-time host monitoring" } else { "container logs" }
                        );
                    } else {
                        tracing::warn!(
                            "⚠️  {} only has {:.2}GB total free (< {:.2}GB required) - FILTERED OUT",
                            backend.name,
                            free_gb,
                            required_gb
                        );
                    }
                }
                
                // If all backends filtered out, check AMD backends too!
                if vram_filtered_candidates.is_empty() && preferred_type == Some(BackendType::Nvidia) {
                    tracing::info!("🔄 All NVIDIA backends full, checking AMD backends...");
                    
                    let amd_candidates: Vec<&discovery::Backend> = reg.iter()
                        .filter(|b| b.backend_type == BackendType::Amd)
                        .collect();
                    
                    for backend in &amd_candidates {
                        // Use same real-time monitoring approach as above
                        let host_gpus = state.gpu_util_state.read().await;
                        
                        let total_free_vram: u64 = if !host_gpus.is_empty() {
                            let gpus = backend.gpus.read().await;
                            if !gpus.is_empty() {
                                gpus.iter()
                                    .filter_map(|container_gpu| {
                                        host_gpus.iter().find(|h| h.gpu_index == container_gpu.device_index && h.backend_type == backend.backend_type)
                                    })
                                    .map(|host_gpu| {
                                        let free_mb = host_gpu.memory_total_mb.saturating_sub(host_gpu.memory_used_mb);
                                        free_mb * 1024 * 1024
                                    })
                                    .sum()
                            } else {
                                if let Some(primary_gpu) = host_gpus.iter().find(|g| g.gpu_index == backend.primary_gpu_index && g.backend_type == backend.backend_type) {
                                    let free_mb = primary_gpu.memory_total_mb.saturating_sub(primary_gpu.memory_used_mb);
                                    free_mb * 1024 * 1024
                                } else {
                                    backend.vram_capacity
                                }
                            }
                        } else {
                            // Fallback to log data
                            let gpus = backend.gpus.read().await;
                            if !gpus.is_empty() {
                                gpus.iter().map(|gpu| gpu.free_memory).sum()
                            } else {
                                backend.vram_capacity * 3
                            }
                        };
                        
                        let free_gb = total_free_vram as f64 / (1024.0 * 1024.0 * 1024.0);
                        
                        if total_free_vram >= required_vram {
                            vram_filtered_candidates.push(*backend);
                            tracing::info!(
                                "✅ AMD fallback: {} has {:.2}GB total free (>= {:.2}GB required)",
                                backend.name, free_gb, required_gb
                            );
                        }
                    }
                }
                
                // If still all backends filtered out, check for AFFINITY FALLBACK
                if vram_filtered_candidates.is_empty() {
                    tracing::warn!(
                        "⚠️ No backend has enough FREE VRAM for model '{}' (requires {:.2}GB). Checking for loaded models (Affinity Fallback)...",
                        model, required_gb
                    );
                    
                    // Fallback: Check if model is already loaded on any candidate
                    let mut affinity_backend = None;
                    
                    for backend in &candidates {
                        let loaded = backend.loaded_models.read().await;
                        // Check if model matches any loaded model
                        if loaded.iter().any(|m| m == model || m == &format!("{}:latest", model)) {
                            affinity_backend = Some(*backend);
                            break;
                        }
                    }
                    
                    if let Some(backend) = affinity_backend {
                        tracing::info!("🎯 Affinity Fallback: Found model '{}' already loaded on {} → Routing request there (Ignoring VRAM check)", model, backend.name);
                        vram_filtered_candidates.push(backend);
                    } else {
                        tracing::error!(
                            "🚫 OOM: No backend has enough VRAM AND model '{}' is not loaded anywhere.",
                            model
                        );
                        return Err(StatusCode::SERVICE_UNAVAILABLE);
                    }
                }
                
                tracing::info!(
                    "📊 Model '{}' requires {:.2}GB → {} backends have sufficient VRAM",
                    model, required_gb, vram_filtered_candidates.len()
                );
                
                // Use filtered candidates for routing
                candidates = vram_filtered_candidates;
            }
        }
    }

    // Load Balancing Logic (now uses VRAM-filtered candidates)
    let mut selected_backend = None;

    if state.model_affinity && model_name.is_some() {
        let model = model_name.as_ref().unwrap();
        tracing::debug!("🔍 Checking model affinity for '{}' across {} backends", model, candidates.len());
        
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
            if loaded {
                tracing::debug!("  ✓ {} has model '{}' loaded (Active: {})", candidates[i].name, model, active);
            }
        }
        
        // Sort: We want Loaded=true first, then lowest active requests
        // Sort by (NOT loaded, active) so loaded=true comes first, then by ascending active
        scored.sort_by_key(|(loaded, active, _)| (!*loaded, *active));
        
        // Take first = backend with model loaded AND lowest active requests
        if let Some((true, _, b)) = scored.first() {
             tracing::info!("🎯 Affinity: ✓ Found model '{}' on {:?} backend {} → routing there", model, b.backend_type, &b.name);
             selected_backend = Some((*b).clone());
        } else {
             tracing::debug!("Model '{}' not loaded on any backend, falling back to {}", model, state.default_strategy);
        }
    }

    if selected_backend.is_none() {
        // Fallback Strategy
        if state.default_strategy == "vram" {
             tracing::debug!("🔍 VRAM Strategy: Using per-GPU tracking across {} backends", reg.len());
             
             // Build physical GPU usage map by aggregating across ALL containers
             let mut physical_gpu_map: std::collections::HashMap<String, (u64, u64, String)> = std::collections::HashMap::new();
             
             let all_backends: Vec<&discovery::Backend> = reg.iter().collect();
             
             // Step 1: Aggregate GPU data from all containers
             for backend in &all_backends {
                 let gpus = backend.gpus.read().await;
                 for gpu in gpus.iter() {
                     let entry = physical_gpu_map.entry(gpu.gpu_id.clone())
                         .or_insert((0, gpu.total_memory, backend.name.clone()));
                     
                     // Calculate usage: sum of (total - free) or layer-based estimate
                     let used = gpu.total_memory.saturating_sub(gpu.free_memory);
                     entry.0 = entry.0.max(used); // Take max usage reported
                     entry.1 = gpu.total_memory;
                 }
             }
             
             if !physical_gpu_map.is_empty() {
                 tracing::debug!("📊 Physical GPU Map: {} unique GPUs detected", physical_gpu_map.len());
                 for (gpu_id, (used, total, backend)) in &physical_gpu_map {
                     let used_gb = *used as f64 / (1024.0 * 1024.0 * 1024.0);
                     let total_gb = *total as f64 / (1024.0 * 1024.0 * 1024.0);
                     let free_gb = total.saturating_sub(*used) as f64 / (1024.0 * 1024.0 * 1024.0);
                     tracing::debug!("  GPU {} (via {}): Used {:.2}GB | Free {:.2}GB | Total {:.2}GB", 
                         &gpu_id[..20.min(gpu_id.len())], backend, used_gb, free_gb, total_gb);
                 }
             }
             
             // Step 2: For each backend, calculate free VRAM on its PRIMARY GPU
             let mut vram_stats: Vec<(u64, &discovery::Backend)> = Vec::new();
             
             for backend in &candidates {
                 let gpus = backend.gpus.read().await;
                 
                 
                 // Sum FREE VRAM across ALL GPUs this container can access
                 // PRIORITY 1: Use visible_gpu_indices + Host Real-time Data (Most accurate & robust)
                 // PRIORITY 2: Use gpus (Log-parsed data) (Fallback if discovery failed to parse env vars)
                 // PRIORITY 3: Use /api/ps (Legacy fallback)
                 
                 let (free_vram, calculation_source) = if !backend.visible_gpu_indices.is_empty() {
                     // 1. Host Monitor Strategy
                     let host_gpus = state.gpu_util_state.read().await;
                     let mut total_free = 0;
                     let mut found_count = 0;
                     
                     if !host_gpus.is_empty() {
                         for &idx in &backend.visible_gpu_indices {
                             if let Some(host_gpu) = host_gpus.iter().find(|g| g.gpu_index == idx && g.backend_type == backend.backend_type) {
                                  // Use host state directly: Total - Used
                                  total_free += (host_gpu.memory_total_mb.saturating_sub(host_gpu.memory_used_mb)) * 1024 * 1024;
                                  found_count += 1;
                             } else {
                                 // Missing host data for a visible GPU? 
                                 // Assume fraction of capacity if we can't see it (better than 0)
                                 if backend.visible_gpu_indices.len() > 0 {
                                    total_free += backend.vram_capacity / backend.visible_gpu_indices.len() as u64;
                                 }
                             }
                         }
                     }
                     
                     // If we found nothing useful in host state (empty?), fall through to others? 
                     // But if indices exist, we should probably stick to this or capacity.
                     if total_free == 0 && found_count == 0 {
                         (backend.vram_capacity, "Static Capacity (Host Data Unavailable)")
                     } else {
                         (total_free, "Host Real-time Monitor")
                     }
                 } else if !gpus.is_empty() {
                     // 2. Log Parsing Strategy (Legacy)
                     let total: u64 = gpus.iter().map(|gpu| {
                          // Look up actual usage in physical GPU map if possible
                         if let Some((used, total, _)) = physical_gpu_map.get(&gpu.gpu_id) {
                             total.saturating_sub(*used)
                         } else {
                             gpu.free_memory
                         }
                     }).sum();
                    (total, "Log Parsing")
                 } else {
                     // 3. API/Static Strategy (Fallback)
                     if let Some(used) = get_used_vram(&state.client, &backend.ip, backend.port).await {
                         let limit = if backend.vram_capacity > 0 {
                             backend.vram_capacity
                         } else {
                             match backend.backend_type {
                                 discovery::BackendType::Nvidia => state.nvidia_vram_limit,
                                 discovery::BackendType::Amd => state.amd_vram_limit,
                             }
                         };
                         (limit.saturating_sub(used), "API/ps Fallback")
                     } else {
                         (0, "Unknown")
                     }
                 };
                 
                 let free_gb = free_vram as f64 / (1024.0 * 1024.0 * 1024.0);
                 
                 // Log backend and per-GPU details (if available) or source
                 tracing::info!("  └─ {:?} backend {}", backend.backend_type, &backend.name);
                 
                 // If we have per-GPU visible indices, log them for clarity
                 if !backend.visible_gpu_indices.is_empty() {
                      // Log specific GPU stats if we have them from host state
                      let host_gpus = state.gpu_util_state.read().await;
                      for &idx in &backend.visible_gpu_indices {
                          if let Some(host_gpu) = host_gpus.iter().find(|g| g.gpu_index == idx && g.backend_type == backend.backend_type) {
                              let free_mb = host_gpu.memory_total_mb.saturating_sub(host_gpu.memory_used_mb);
                              tracing::info!("     GPU {}: {:.2}GB free (Host Monitor)", idx, free_mb as f64 / 1024.0);
                          }
                      }
                 } else if !gpus.is_empty() {
                     // Fallback to logging internal GPU structs if we used that path
                     for gpu in gpus.iter() {
                         let gpu_free = if let Some((used, total, _)) = physical_gpu_map.get(&gpu.gpu_id) {
                             total.saturating_sub(*used)
                         } else {
                             gpu.free_memory
                         };
                         let gpu_free_gb = gpu_free as f64 / (1024.0 * 1024.0 * 1024.0);
                         tracing::info!("     GPU {}: {:.2}GB free (Log Data)", gpu.device_index, gpu_free_gb);
                     }
                 }
                 
                 tracing::info!("     → Total: {:.2}GB free (Source: {})", free_gb, calculation_source);
                 
                 vram_stats.push((free_vram, backend));
             }
             
             // Sort by Free VRAM Descending
             vram_stats.sort_by_key(|(free, _)| Reverse(*free));
             
             if let Some((max_free, _)) = vram_stats.first() {
                  // Get all backends with the maximum free VRAM (for tie-breaking)
                  let max_candidates: Vec<&discovery::Backend> = vram_stats.iter()
                      .filter(|(free, _)| free == max_free)
                      .map(|(_, b)| *b)
                      .collect();
                  
                  // Randomly select among tied candidates
                  if let Some(selected) = max_candidates.choose(&mut rand::thread_rng()) {
                      tracing::info!(
                          "VRAM Strategy: ✓ Selected {:?} backend {} with {:.2}GB free (from {} candidates, {} tied at max)",
                          selected.backend_type,
                          &selected.name,
                          *max_free as f64 / (1024.0 * 1024.0 * 1024.0),
                          vram_stats.len(),
                          max_candidates.len()
                      );
                      selected_backend = Some((*selected).clone());
                  }
             }
        } else if state.default_strategy == "round_robin" {
            // Round-robin across ALL backends
            let all_backends: Vec<&discovery::Backend> = reg.iter().collect();
            if !all_backends.is_empty() {
                let idx = state.round_robin_counter.fetch_add(1, Ordering::Relaxed) % all_backends.len();
                selected_backend = Some(all_backends[idx].clone());
                if let Some(ref b) = selected_backend {
                    tracing::info!(
                        "Round Robin: ✓ Selected {:?} backend {} (#{} of {})",
                        b.backend_type,
                        &b.name,
                        idx + 1,
                        all_backends.len()
                    );
                }
            }
        } else if state.default_strategy == "nvidia" {
            // Force NVIDIA only backends
            let nvidia_backends: Vec<&discovery::Backend> = reg.iter()
                .filter(|b| b.backend_type == BackendType::Nvidia)
                .collect();
            if !nvidia_backends.is_empty() {
                selected_backend = nvidia_backends.choose(&mut rand::thread_rng()).cloned().map(|b| b.clone());
                if let Some(ref b) = selected_backend {
                    tracing::info!("NVIDIA Strategy: ✓ Selected {} ({} NVIDIA backends available)", &b.name, nvidia_backends.len());
                }
            } else {
                tracing::warn!("NVIDIA Strategy: No NVIDIA backends available!");
            }
        } else if state.default_strategy == "amd" {
            // Force AMD only backends
            let amd_backends: Vec<&discovery::Backend> = reg.iter()
                .filter(|b| b.backend_type == BackendType::Amd)
                .collect();
            if !amd_backends.is_empty() {
                selected_backend = amd_backends.choose(&mut rand::thread_rng()).cloned().map(|b| b.clone());
                if let Some(ref b) = selected_backend {
                    tracing::info!("AMD Strategy: ✓ Selected {} ({} AMD backends available)", &b.name, amd_backends.len());
                }
            } else {
                tracing::warn!("AMD Strategy: No AMD backends available!");
            }
        } else if state.default_strategy == "alternate" {
            
            let last_type = state.last_backend_type.load(Ordering::Relaxed);
            // Determine next type to use
            let next_type = if last_type == 0 {
                BackendType::Amd
            } else {
                BackendType::Nvidia
            };
            
            tracing::debug!("🔄 Alternate Strategy: Last used {:?}, switching to {:?}", 
                if last_type == 0 { "Nvidia" } else { "Amd" }, 
                next_type
            );
            
            // Get all backends of the next type
            let mut next_candidates: Vec<&discovery::Backend> = reg.iter()
                .filter(|b| b.backend_type == next_type)
                .collect();
            
            // Sort by name for deterministic ordering
            next_candidates.sort_by_key(|b| &b.name);
            
            // Select a random one from the next type (for load distribution within the type)
            if !next_candidates.is_empty() {
                selected_backend = next_candidates.choose(&mut rand::thread_rng()).cloned().map(|b| b.clone());
                if let Some(ref b) = selected_backend {
                    tracing::info!(
                        "Alternate Strategy: ✓ Switching to {:?} backend {} ({} available)",
                        next_type,
                        &b.name,
                        next_candidates.len()
                    );
                }
            } else {
                // Fallback: if no backends of next type, use any available from candidates
                tracing::warn!("Alternate Strategy: No {:?} backends available, falling back to random", next_type);
                selected_backend = candidates.choose(&mut rand::thread_rng()).cloned().map(|b| b.clone());
            }
        }
        
        // If still none (or strategy != vram && != alternate), use Random
        if selected_backend.is_none() {
            selected_backend = candidates.choose(&mut rand::thread_rng()).cloned().map(|b| b.clone());
            if let Some(ref b) = selected_backend {
                tracing::info!("Fallback: ✓ Random selection → {:?} backend {}", b.backend_type, &b.name);
            }
        }
    }

    let target_backend = match selected_backend {
        Some(b) => {
            // If ENFORCE_STRATEGY=false, check if backend has enough free VRAM to prevent OOM
            // This works regardless of MODEL_AFFINITY setting
            if !state.enforce_strategy && model_name.is_some() {
                let model = model_name.as_ref().unwrap();
                
                // Check if selected backend has sufficient free VRAM (use basic heuristic: >2GB free)
                // This prevents OOM errors when VRAM calculation is inaccurate
                if let Some(used) = get_used_vram(&state.client, &b.ip, b.port).await {
                    let limit = if b.vram_capacity > 0 {
                        b.vram_capacity
                    } else {
                        match b.backend_type {
                            BackendType::Nvidia => state.nvidia_vram_limit,
                            BackendType::Amd => state.amd_vram_limit,
                        }
                    };
                    
                    let free = if limit > used { limit - used } else { 0 };
                    let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
                    
                    // If less than 2GB free, fall back to affinity
                    if free_gb < 2.0 {
                        tracing::warn!(
                            "⚠️  Selected backend {} has only {:.2}GB free (may be insufficient)",
                            &b.name, free_gb
                        );
                        
                        // Check all backends for ones with model already loaded
                        let mut checks = Vec::new();
                        let all_backends: Vec<&discovery::Backend> = reg.iter().collect();
                        for backend in &all_backends {
                            checks.push(check_if_model_loaded(&state.client, &backend.ip, backend.port, model));
                        }
                        
                        let results = join_all(checks).await;
                        let mut backends_with_model = Vec::new();
                        
                        for (i, is_loaded) in results.into_iter().enumerate() {
                            if is_loaded.unwrap_or(false) {
                                backends_with_model.push(all_backends[i]);
                            }
                        }
                        
                        if !backends_with_model.is_empty() {
                            // Sort by name for deterministic ordering
                            backends_with_model.sort_by_key(|b| &b.name);
                            
                            // Round-robin among backends with model loaded
                            let idx = state.round_robin_counter.fetch_add(1, Ordering::Relaxed) % backends_with_model.len();
                            let fallback = backends_with_model[idx];
                            
                            tracing::info!(
                                "🔄 Fallback: Routing to {:?} backend {} which has '{}' loaded (#{} of {})",
                                fallback.backend_type,
                                &fallback.name,
                                model,
                                idx + 1,
                                backends_with_model.len()
                            );
                            
                            // Update last used backend type for alternate strategy
                            let type_val = match fallback.backend_type {
                                BackendType::Nvidia => 0,
                                BackendType::Amd => 1,
                            };
                            state.last_backend_type.store(type_val, Ordering::Relaxed);
                            
                            fallback.clone()
                        } else {
                            tracing::warn!("⚠️  No backend has '{}' loaded, proceeding with low-VRAM backend (may OOM)", model);
                            // Update last used backend type
                            let type_val = match b.backend_type {
                                BackendType::Nvidia => 0,
                                BackendType::Amd => 1,
                            };
                            state.last_backend_type.store(type_val, Ordering::Relaxed);
                            b
                        }
                    } else {
                        // Sufficient VRAM, proceed with selected backend
                        let type_val = match b.backend_type {
                            BackendType::Nvidia => 0,
                            BackendType::Amd => 1,
                        };
                        state.last_backend_type.store(type_val, Ordering::Relaxed);
                        b
                    }
                } else {
                    // Failed to check VRAM, proceed anyway
                    let type_val = match b.backend_type {
                        BackendType::Nvidia => 0,
                        BackendType::Amd => 1,
                    };
                    state.last_backend_type.store(type_val, Ordering::Relaxed);
                    b
                }
            } else {
                // ENFORCE_STRATEGY=true or no model name, proceed with selected backend
                let type_val = match b.backend_type {
                    BackendType::Nvidia => 0,
                    BackendType::Amd => 1,
                };
                state.last_backend_type.store(type_val, Ordering::Relaxed);
                b
            }
        },
        None => return Err(StatusCode::SERVICE_UNAVAILABLE),
    };

    let target_url = format!("http://{}:{}{}", target_backend.ip, target_backend.port, path);
    tracing::info!(
        "→ Proxying {} {} to {:?} backend {} | {} (Active: {})",
        method,
        path,
        target_backend.backend_type,
        &target_backend.name,
        target_url,
        target_backend.active_requests.load(Ordering::Relaxed)
    );

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
    // DEPRECATED: /api/ps approach - it counts models across ALL GPUs
    // We need to query the backend's GPU directly for accurate primary GPU usage
    // This requires Docker API access which we don't have in the main load balancer
    
    // Fallback: Use /api/ps but this will be inaccurate for multi-GPU setups
    // TODO: This should be moved to discovery.rs where we have Docker access
    let url = format!("http://{}:{}/api/ps", ip, port);
    match client.get(&url).timeout(std::time::Duration::from_millis(500)).send().await {
         Ok(res) => {
             if let Ok(json) = res.json::<Value>().await {
                 let mut used = 0;
                 if let Some(models) = json["models"].as_array() {
                      for m in models {
                           // WARNING: This sums ALL models across ALL visible GPUs
                           // In multi-GPU setups, this will overcount
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

// Returns the request (possibly modified), preferred backend type, model name, AND targeted indices
async fn determine_intent(req: Request) -> (Request, Option<BackendType>, Option<String>, Option<Vec<usize>>) {
    let (mut parts, body) = req.into_parts();
    let mut intent: Option<BackendType> = None;  // No default preference - check all backends 
    let mut found_model = None;
    let mut target_indices = None;
    
    // Only parse body for API methods that send JSON
    let path = parts.uri.path();
    if !path.starts_with("/api/generate") && !path.starts_with("/api/chat") && !path.starts_with("/api/show") && !path.starts_with("/api/pull") {
         let req = Request::from_parts(parts, body);
         return (req, intent, None, None);
    }

    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(_) => return (Request::from_parts(parts, Body::empty()), intent, None, None),
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
                
                // Helper to parse "amd:0,2" -> (Amd, Some([0, 2]))
                let parse_suffix = |suffix: &str| -> Option<(BackendType, Option<Vec<usize>>)> {
                    if suffix == "nvidia" { return Some((BackendType::Nvidia, None)); }
                    if suffix == "amd" { return Some((BackendType::Amd, None)); }
                    
                    if let Some(rest) = suffix.strip_prefix("nvidia:") {
                        let idxs: Vec<usize> = rest.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                        if !idxs.is_empty() { return Some((BackendType::Nvidia, Some(idxs))); }
                    }
                    if let Some(rest) = suffix.strip_prefix("amd:") {
                        let idxs: Vec<usize> = rest.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                        if !idxs.is_empty() { return Some((BackendType::Amd, Some(idxs))); }
                    }
                    None
                };

                // Prefix checks (Simple)
                if let Some(stripped) = model.strip_prefix("nvidia/") {
                    intent = Some(BackendType::Nvidia); new_model = stripped.to_string(); matched = true;
                } else if let Some(stripped) = model.strip_prefix("cuda/") {
                    intent = Some(BackendType::Nvidia); new_model = stripped.to_string(); matched = true;
                } else if let Some(stripped) = model.strip_prefix("amd/") {
                    intent = Some(BackendType::Amd); new_model = stripped.to_string(); matched = true;
                } else if let Some(stripped) = model.strip_prefix("rocm/") {
                     intent = Some(BackendType::Amd); new_model = stripped.to_string(); matched = true;
                } 
                else if let Some((base, suffix)) = model.rsplit_once('@') {
                    if let Some((b_type, indices)) = parse_suffix(suffix) {
                        intent = Some(b_type);
                        target_indices = indices;
                        new_model = base.to_string();
                        matched = true;
                    }
                }
                
                // Set found model and normalize: add ":latest" if no tag specified
                let normalized_model = if new_model.contains(':') {
                    new_model.clone()
                } else {
                    format!("{}:latest", new_model)
                };
                found_model = Some(normalized_model.clone());

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
    (req, intent, found_model, target_indices)
}

// Query model size and calculate VRAM requirements
async fn get_model_vram_requirement(
    client: &Client,
    backend: &discovery::Backend,
    model_name: &str
) -> Option<u64> {
    let url = format!("http://{}:{}/api/tags", backend.ip, backend.port);
    
    match client.get(&url).timeout(Duration::from_secs(2)).send().await {
        Ok(resp) => {
            if let Ok(json) = resp.json::<Value>().await {
                if let Some(models) = json.get("models").and_then(|m| m.as_array()) {
                    // Find the requested model
                    for model in models {
                        if let Some(name) = model.get("name").and_then(|n| n.as_str()) {
                            if name == model_name || name.starts_with(&format!("{}:", model_name)) {
                                // Extract model size
                                if let Some(size) = model.get("size").and_then(|s| s.as_u64()) {
                                    // Calculate required VRAM:
                                    // Model size + 10% overhead for KV cache
                                    let overhead = (size as f64 * 0.10) as u64;
                                    let required = size + overhead;
                                    
                                    tracing::debug!(
                                        "📏 Model '{}' size={:.2}GB + 10% overhead={:.2}GB → required={:.2}GB",
                                        model_name,
                                        size as f64 / (1024.0 * 1024.0 * 1024.0),
                                        overhead as f64 / (1024.0 * 1024.0 * 1024.0),
                                        required as f64 / (1024.0 * 1024.0 * 1024.0)
                                    );
                                    
                                    return Some(required);
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            tracing::warn!("Failed to query model size from {}: {}", backend.name, e);
        }
    }
    
    None
}
