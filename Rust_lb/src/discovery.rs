use bollard::Docker;
use bollard::container::ListContainersOptions;
use bollard::container::LogsOptions;
use bollard::exec::{CreateExecOptions, StartExecResults};
use bollard::models::ContainerSummary;
use futures_util::stream::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;
use tracing::{info, error, warn};


#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BackendType {
    Nvidia,
    Amd,
}

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub gpu_id: String,              // e.g., "GPU-575cc87d-0d69-3626-86fd-ac7a93e687e3"
    pub gpu_name: String,             // e.g., "CUDA1" or "CUDA0"
    pub device_index: usize,          // Physical device index from container's perspective
    pub layers_loaded: HashMap<String, usize>, // model_name -> layer_count
    pub free_memory: u64,             // Bytes
    pub total_memory: u64,            // Bytes
}

#[derive(Clone, Debug)]
pub struct Backend {
    pub id: String,
    pub name: String,
    pub ip: String,
    pub port: u16,
    pub backend_type: BackendType,
    pub active_requests: Arc<AtomicUsize>,
    pub vram_capacity: u64,           // Total capacity (for fallback)
    pub gpus: Arc<RwLock<Vec<GpuInfo>>>, // Per-GPU tracking
    pub primary_gpu_index: usize,     // Index of primary GPU (first in CUDA_VISIBLE_DEVICES)
    pub visible_gpu_indices: Vec<usize>, // All visible GPU indices
    pub loaded_models: Arc<RwLock<Vec<String>>>, // List of model names currently loaded
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
    
    // Start log streaming for each backend
    {
        let reg = registry.read().await;
        for backend in reg.iter() {
            let docker_clone = docker.clone();
            let backend_name = backend.name.clone();
            let container_id = backend.id.clone();
            let gpus = backend.gpus.clone();
            
            tokio::spawn(async move {
                stream_container_logs(&docker_clone, &container_id, backend_name, gpus).await;
            });
        }
    }

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
                    let (counter, vram, loaded) = if let Some(existing) = existing_map.get(&id) {
                        // Reuse
                        (existing.active_requests.clone(), existing.vram_capacity, existing.loaded_models.clone())
                    } else {
                        // New backend, init counter and DETECT VRAM
                        let c = Arc::new(AtomicUsize::new(0));
                        let v = autodetect_vram(docker, &id, &b_type).await;
                        let l = Arc::new(RwLock::new(Vec::new()));
                        (c, v, l)
                    };

                    // Still check running state
                    if container.state.as_deref() == Some("running") {
                         // Extract name (strip leading /)
                         let name = container.names.as_ref()
                             .and_then(|n| n.first())
                             .map(|s| s.trim_start_matches('/').to_string())
                             .unwrap_or_else(|| id.clone());

                         // Inspect container to get environment variables
                         let (primary_gpu, visible_gpus) = match docker.inspect_container(&id, None).await {
                            Ok(inspect) => {
                                if let Some(config) = inspect.config {
                                    if let Some(env) = config.env {
                                        parse_visible_gpu_indices(&env)
                                    } else { (0, vec![0]) }
                                } else { (0, vec![0]) }
                            }
                            Err(_) => (0, vec![0])
                        };

                         new_backends.push(Backend {
                             id,
                             name,
                             ip,
                             port: 11434,
                             backend_type: b_type,
                             active_requests: counter,
                             vram_capacity: vram,
                             gpus: Arc::new(RwLock::new(Vec::new())), // Will be populated by log parsing
                             primary_gpu_index: primary_gpu,
                             visible_gpu_indices: visible_gpus,
                             loaded_models: loaded,
                         });
                    }
                }
            }
            
            let mut reg = registry.write().await;
            *reg = new_backends.clone();
            info!("Updated backend registry: {} instances found.", reg.len());
            for b in reg.iter() {
                let active = b.active_requests.load(Ordering::Relaxed);
                info!("  - [{:?}] {} @ {}:{} (Active: {}, VRAM: {} MB)", b.backend_type, b.name, b.ip, b.port, active, b.vram_capacity / (1024*1024));
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
    // 1. Inspect container to get Env vars (for AMD visibility)
    let env_vars = if let Ok(info) = docker.inspect_container(container_id, None).await {
        info.config.and_then(|c| c.env).unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut visible_indices: Option<Vec<usize>> = None;
    if matches!(b_type, BackendType::Amd) {
        // Check HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES
        for var in &env_vars {
            if var.starts_with("HIP_VISIBLE_DEVICES=") || var.starts_with("ROCR_VISIBLE_DEVICES=") {
                if let Some(val) = var.split('=').nth(1) {
                    let idxs: Vec<usize> = val.split(',')
                        .map(|s| s.trim())
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    visible_indices = Some(idxs);
                    break; 
                }
            }
        }
    }

    let cmd = match b_type {
        BackendType::Nvidia => vec!["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        BackendType::Amd => vec![
            "sh", "-c", 
            // Check driver link AND read VRAM if amdgpu, skip non-AMD cards
            "for d in /sys/class/drm/card*; do [ -L $d/device/driver ] && readlink $d/device/driver | grep -q amdgpu && [ -f $d/device/mem_info_vram_total ] && cat $d/device/mem_info_vram_total; done"
        ],
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
                    // Return ONLY the first GPU's capacity (primary GPU)
                    for line in full_output.lines() {
                        if let Ok(val) = line.trim().parse::<u64>() {
                            // First GPU in list is the primary
                            return val * 1024 * 1024;
                        }
                    }
                },
                BackendType::Amd => {
                    // Collect all physical AMD card sizes
                    let mut card_sizes = Vec::new();
                    for line in full_output.lines() {
                        if let Ok(val) = line.trim().parse::<u64>() {
                            card_sizes.push(val);
                        }
                    }
                    
                    // Return ONLY the primary GPU's capacity (first visible one)
                    if let Some(idxs) = visible_indices {
                        // First index in ROCR_VISIBLE_DEVICES is the primary GPU
                        if let Some(&first_idx) = idxs.first() {
                            if let Some(size) = card_sizes.get(first_idx) {
                                return *size;
                            }
                        }
                    } else {
                        // If no env var, primary is first physical GPU
                        if let Some(size) = card_sizes.first() {
                            return *size;
                        }
                    }
                }
            }
        }
    }
    0
}

// ============================================================================
// Log Streaming and Parsing for Per-GPU VRAM Tracking
// ============================================================================

async fn stream_container_logs(
    docker: &Docker,
    container_id: &str,
    backend_name: String,
    gpus: Arc<RwLock<Vec<GpuInfo>>>
) {
    info!("🔍 Starting log stream for backend: {}", backend_name);
    
    let options = LogsOptions::<String> {
        follow: true,
        stdout: true,
        stderr: true,
        timestamps: false,
        since: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64,
        ..Default::default()
    };
    
    let mut log_stream = docker.logs(container_id, Some(options));
    
    while let Some(log_result) = log_stream.next().await {
        match log_result {
            Ok(output) => {
                let log_line = output.to_string();
                parse_ollama_log_line(&log_line, &gpus, &backend_name).await;
            }
            Err(e) => {
                warn!("Log stream error for {}: {}", backend_name, e);
                break;
            }
        }
    }
    warn!("Log stream ended for {}", backend_name);
}

async fn parse_ollama_log_line(
    line: &str,
    gpus: &Arc<RwLock<Vec<GpuInfo>>>,
    backend_name: &str
) {
    // Pattern 1: GPU Discovery
    // Example: msg="inference compute" id=GPU-575cc87d-0d69-3626-86fd-ac7a93e687e3 name=CUDA1 total="8.0 GiB"
    if line.contains("inference compute") && line.contains("id=GPU-") {
        if let Some(gpu_id_start) = line.find("id=GPU-") {
            let id_substr = &line[gpu_id_start + 3..];
            if let Some(gpu_id_end) = id_substr.find(|c: char| c.is_whitespace()) {
                let gpu_id = id_substr[..gpu_id_end].to_string();
                
                // Extract GPU name (CUDA0, CUDA1, etc.)
                let gpu_name = if let Some(name_start) = line.find("name=") {
                    let name_substr = &line[name_start + 5..];
                    if let Some(name_end) = name_substr.find(|c: char| c.is_whitespace()) {
                        name_substr[..name_end].to_string()
                    } else {
                        "UNKNOWN".to_string()
                    }
                } else {
                    "UNKNOWN".to_string()
                };
                
                // Extract total memory
                let total_bytes = if let Some(total_start) = line.find("total=\"") {
                    let total_substr = &line[total_start + 7..];
                    if let Some(total_end) = total_substr.find(" GiB\"") {
                        let gib: f64 = total_substr[..total_end].parse().unwrap_or(0.0);
                        (gib * 1024.0 * 1024.0 * 1024.0) as u64
                    } else {
                        0
                    }
                } else {
                    0
                };
                
                if total_bytes > 0 {
                    let mut gpus_vec = gpus.write().await;
                    
                    // Check if GPU already exists
                    if !gpus_vec.iter().any(|g| g.gpu_id == gpu_id) {
                        // Extract device index from name (CUDA0 -> 0, CUDA1 -> 1, HIP0 -> 0, HIP1 -> 1)
                        let device_index = if gpu_name.starts_with("CUDA") {
                            gpu_name.trim_start_matches("CUDA").parse().unwrap_or(0)
                        } else if gpu_name.starts_with("HIP") {
                            gpu_name.trim_start_matches("HIP").parse().unwrap_or(0)
                        } else {
                            // Unknown format, try to extract any trailing digits
                            gpu_name.chars()
                                .rev()
                                .take_while(|c| c.is_numeric())
                                .collect::<String>()
                                .chars()
                                .rev()
                                .collect::<String>()
                                .parse()
                                .unwrap_or(0)
                        };
                        
                        gpus_vec.push(GpuInfo {
                            gpu_id: gpu_id.clone(),
                            gpu_name: gpu_name.clone(),
                            device_index,
                            layers_loaded: HashMap::new(),
                            free_memory: total_bytes,
                            total_memory: total_bytes,
                        });
                        
                        info!("✨ {}: Discovered GPU {} ({}) with {:.2} GiB", 
                            backend_name, gpu_name, &gpu_id[..20], total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
                    }
                }
            }
        }
    }
    
    // Pattern 2: GPU Memory Updates
    // Example: msg="gpu memory" id=GPU-575cc87d-0d69-3626-86fd-ac7a93e687e3 free="7.8 GiB"
    if line.contains("gpu memory") && line.contains("id=GPU-") && line.contains("free=\"") {
        if let Some(gpu_id_start) = line.find("id=GPU-") {
            let id_substr = &line[gpu_id_start + 3..];
            if let Some(gpu_id_end) = id_substr.find(|c: char| c.is_whitespace()) {
                let gpu_id = id_substr[..gpu_id_end].to_string();
                
                // Extract free memory
                if let Some(free_start) = line.find("free=\"") {
                    let free_substr = &line[free_start + 6..];
                    if let Some(free_end) = free_substr.find(" GiB\"") {
                        let gib: f64 = free_substr[..free_end].parse().unwrap_or(0.0);
                        let free_bytes = (gib * 1024.0 * 1024.0 * 1024.0) as u64;
                        
                        let mut gpus_vec = gpus.write().await;
                        if let Some(gpu) = gpus_vec.iter_mut().find(|g| g.gpu_id == gpu_id) {
                            gpu.free_memory = free_bytes;
                        }
                    }
                }
            }
        }
    }
    
    // Pattern 3: Layer Allocation
    // Example: GPULayers:41[ID:GPU-575cc87d... Layers:21(0..20) ID:GPU-33b0b... Layers:20(21..40)]
    if line.contains("GPULayers:") && line.contains("ID:GPU-") && line.contains("Layers:") {
        // Extract model identifier (use hash from earlier in line if available)
        let model_name = extract_model_identifier(line);
        
        // Find all GPU layer allocations in this line
        let mut pos = 0;
        while let Some(id_start) = line[pos..].find("ID:GPU-") {
            let abs_start = pos + id_start + 3; // Position of "GPU-"
            let id_substr = &line[abs_start..];
            
            if let Some(id_end) = id_substr.find(|c: char| c.is_whitespace()) {
                let gpu_id = id_substr[..id_end].to_string();
                
                // Find "Layers:" after this GPU ID
                if let Some(layers_start) = line[abs_start..].find(" Layers:") {
                    let layers_pos = abs_start + layers_start + 8;
                    let layers_substr = &line[layers_pos..];
                    
                    if let Some(layers_end) = layers_substr.find('(') {
                        if let Ok(layer_count) = layers_substr[..layers_end].parse::<usize>() {
                            let mut gpus_vec = gpus.write().await;
                            if let Some(gpu) = gpus_vec.iter_mut().find(|g| g.gpu_id == gpu_id) {
                                *gpu.layers_loaded.entry(model_name.clone()).or_insert(0) = layer_count;
                                info!("📊 {}: GPU {} allocated {} layers for {}", 
                                    backend_name, gpu.gpu_name, layer_count, model_name);
                            }
                        }
                    }
                }
                
                pos = abs_start + id_end;
            } else {
                break;
            }
        }
    }
}

fn extract_model_identifier(line: &str) -> String {
    // Try to extract SHA256 hash from model path
    if let Some(idx) = line.find("sha256-") {
        if let Some(hash_part) = line[idx + 7..].split(|c: char| !c.is_alphanumeric()).next() {
            if hash_part.len() >= 12 {
                return format!("model_{}", &hash_part[..12]);
            }
        }
    }
    
    // Fallback: use timestamp
    format!("model_{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs())
}

// Parse visible GPU indices from container environment variables 
fn parse_visible_gpu_indices(env: &[String]) -> (usize, Vec<usize>) {
    for var in env {
        // NVIDIA: CUDA_VISIBLE_DEVICES=0,1,2
        if var.starts_with("CUDA_VISIBLE_DEVICES=") {
            if let Some(devices) = var.strip_prefix("CUDA_VISIBLE_DEVICES=") {
                let indices: Vec<usize> = devices.split(',')
                    .map(|s| s.trim())
                    .filter_map(|s| s.parse().ok())
                    .collect();
                
                if let Some(&first) = indices.first() {
                    return (first, indices);
                }
            }
        }
        // AMD: ROCR_VISIBLE_DEVICES=1,0
        if var.starts_with("ROCR_VISIBLE_DEVICES=") {
            if let Some(devices) = var.strip_prefix("ROCR_VISIBLE_DEVICES=") {
                let indices: Vec<usize> = devices.split(',')
                    .map(|s| s.trim())
                    .filter_map(|s| s.parse().ok())
                    .collect();
                
                if let Some(&first) = indices.first() {
                    return (first, indices);
                }
            }
        }
    }
    (0, vec![0]) // Default to GPU 0
}
