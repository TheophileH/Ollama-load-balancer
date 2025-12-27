use bollard::Docker;
use bollard::container::ListContainersOptions;
use bollard::exec::{CreateExecOptions, StartExecResults};
use bollard::models::ContainerSummary;
use futures_util::stream::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;
use tracing::{info, error, warn};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub enum BackendType {
    Nvidia,
    Amd,
}

#[derive(Clone, Debug)]
pub struct Backend {
    pub id: String,
    pub ip: String,
    pub port: u16,
    pub backend_type: BackendType,
    pub active_requests: Arc<AtomicUsize>,
    pub vram_capacity: u64,
}

pub type BackendRegistry = Arc<RwLock<Vec<Backend>>>;

pub async fn start_discovery(registry: BackendRegistry) {
    let docker = match Docker::connect_with_socket_defaults() {
        Ok(d) => d,
        Err(e) => {
            error!("Failed to connect to Docker socket: {}", e);
            return;
        }
    };

    // Initial scan
    update_backends(&docker, &registry).await;

    // Watch loop
    let docker_clone = docker.clone();
    let registry_clone = registry.clone();
    tokio::spawn(async move {
        let mut event_stream = docker_clone.events::<String>(None);
        while let Some(event) = event_stream.next().await {
            match event {
                Ok(evt) => {
                    if let Some(action) = evt.action {
                        if action == "start" || action == "die" || action == "stop" {
                             update_backends(&docker_clone, &registry_clone).await;
                        }
                    }
                }
                Err(e) => error!("Docker event stream error: {}", e),
            }
        }
    });
}

async fn update_backends(docker: &Docker, registry: &BackendRegistry) {
    let mut filters = HashMap::new();
    filters.insert("label", vec!["ollama.backend=true"]);

    let options = ListContainersOptions {
        all: true, 
        filters,
        ..Default::default()
    };

    // Capture existing backends to preserve state (Requests + VRAM)
    let existing_map: HashMap<String, Backend> = {
        let reg = registry.read().await;
        reg.iter()
            .map(|b| (b.id.clone(), b.clone()))
            .collect()
    };

    match docker.list_containers(Some(options)).await {
        Ok(containers) => {
            let mut new_backends = Vec::new();
            for container in containers {
                let id = container.id.clone().unwrap_or_default();
                
                // Parse basic info first
                if let Some((ip, b_type)) = parse_basic_info(&container) {
                    // Check existence
                    let (counter, vram) = if let Some(existing) = existing_map.get(&id) {
                        // Reuse
                        (existing.active_requests.clone(), existing.vram_capacity)
                    } else {
                        // New backend, init counter and DETECT VRAM
                        let c = Arc::new(AtomicUsize::new(0));
                        let v = autodetect_vram(docker, &id, &b_type).await;
                        (c, v)
                    };

                    // Still check running state
                    if container.state.as_deref() == Some("running") {
                         new_backends.push(Backend {
                             id,
                             ip,
                             port: 11434,
                             backend_type: b_type,
                             active_requests: counter,
                             vram_capacity: vram,
                         });
                    }
                }
            }
            
            let mut reg = registry.write().await;
            *reg = new_backends.clone();
            info!("Updated backend registry: {} instances found.", reg.len());
            for b in reg.iter() {
                let active = b.active_requests.load(Ordering::Relaxed);
                info!("  - [{:?}] {} @ {}:{} (Active: {}, VRAM: {} MB)", b.backend_type, b.id, b.ip, b.port, active, b.vram_capacity / (1024*1024));
            }
        },
        Err(e) => error!("Failed to list containers: {}", e),
    }
}

fn parse_basic_info(c: &ContainerSummary) -> Option<(String, BackendType)> {
    let labels = c.labels.as_ref()?;
    let backend_type_str = labels.get("ollama.type")?;
    
    let backend_type = match backend_type_str.as_str() {
        "nvidia" => BackendType::Nvidia,
        "amd" => BackendType::Amd,
        _ => return None,
    };

    let network_settings = c.network_settings.as_ref()?;
    let networks = network_settings.networks.as_ref()?;
    let ip = networks.values().next()?.ip_address.as_ref()?.clone();
    if ip.is_empty() { return None; }

    Some((ip, backend_type))
}

async fn autodetect_vram(docker: &Docker, container_id: &str, b_type: &BackendType) -> u64 {
    let cmd = match b_type {
        BackendType::Nvidia => vec!["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        BackendType::Amd => vec!["rocm-smi", "--showvram", "--json"], // JSON parsing requires serde logic, let's keep it simple text if possible? 
        // rocm-smi output is complex.
        // Let's assume user configured env var for AMD if rocm-smi is hard.
        // Or we can try basic parsing.
    };

    let exec_config = CreateExecOptions {
        attach_stdout: Some(true),
        cmd: Some(cmd),
        ..Default::default()
    };

    if let Ok(exec) = docker.create_exec(container_id, exec_config).await {
        if let Ok(StartExecResults::Attached { mut output, .. }) = docker.start_exec(&exec.id, None).await {
            // Collect output
            let mut full_output = String::new();
            while let Some(Ok(msg)) = output.next().await {
                 full_output.push_str(&msg.to_string());
            }
            
            // Parse
            match b_type {
                BackendType::Nvidia => {
                    // Sum lines
                    let mut total = 0;
                    for line in full_output.lines() {
                        if let Ok(val) = line.trim().parse::<u64>() {
                            total += val;
                        }
                    }
                    if total > 0 { return total * 1024 * 1024; }
                },
                BackendType::Amd => {
                    // Quick hack for AMD: If rocm-smi fails or returns complex json
                    // We might skip complex JSON parsing here to avoid heavy deps inside 'detect' logic
                    // unless we use serde_json.
                    // Let's assume standard rocm-smi textual output if json fails or use regex?
                    // Actually, if we use JSON we can use serde_json.
                    if let Ok(json) = serde_json::from_str::<Value>(&full_output) {
                         if let Some(obj) = json.as_object() {
                             let mut total = 0;
                             for (_card, data) in obj {
                                 if let Some(bytes_str) = data.get("VRAM Total Memory (B)").and_then(|v| v.as_str()) {
                                     if let Ok(bytes) = bytes_str.parse::<u64>() {
                                         total += bytes;
                                     }
                                 }
                             }
                             if total > 0 { return total; }
                         }
                    }
                }
            }
        }
    }
    0
}
