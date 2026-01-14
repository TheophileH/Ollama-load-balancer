use crate::discovery::{Backend, BackendRegistry, GpuInfo};
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{info, warn, debug};

/// VRAM Monitor - Dedicated thread for polling backend VRAM status
/// Runs independently from log parsing to ensure real-time updates

pub async fn start_vram_monitor(
    client: Client,
    registry: BackendRegistry,
    poll_interval_secs: u64
) {
    info!("🔄 Starting VRAM monitor (polling every {}s)", poll_interval_secs);
    
    let mut interval = tokio::time::interval(Duration::from_secs(poll_interval_secs));
    
    loop {
        interval.tick().await;
        
        // Poll all backends
        let backends = {
            let reg = registry.read().await;
            reg.clone()
        };
        
        if backends.is_empty() {
            debug!("No backends to monitor, skipping poll");
            continue;
        }
        
        debug!("📊 Polling VRAM status from {} backends", backends.len());
        
        for backend in &backends {
            poll_backend_vram(&client, backend).await;
        }
    }
}

async fn poll_backend_vram(client: &Client, backend: &Backend) {
    let url = format!("http://{}:{}/api/ps", backend.ip, backend.port);
    
    match client.get(&url).timeout(Duration::from_secs(2)).send().await {
        Ok(resp) => {
            if let Ok(json) = resp.json::<Value>().await {
                if let Some(models) = json.get("models").and_then(|m| m.as_array()) {
                    update_gpu_info_from_api_ps(backend, models).await;
                } else {
                    debug!("{}: No models loaded", backend.name);
                    // Clear GPU usage if no models
                    clear_gpu_usage(backend).await;
                }
            }
        }
        Err(e) => {
            warn!("{}: Failed to poll VRAM: {}", backend.name, e);
        }
    }
}

async fn update_gpu_info_from_api_ps(backend: &Backend, models: &[Value]) {
    let mut gpus = backend.gpus.write().await;
    
    // If we don't have GPU info yet, we can't update properly
    if gpus.is_empty() {
        // Try to infer from model data
        let mut total_vram_used: u64 = 0;
        
        for model in models {
            if let Some(size_vram) = model.get("size_vram").and_then(|s| s.as_u64()) {
                total_vram_used += size_vram;
            }
        }
        
        if total_vram_used > 0 {
            // Create a placeholder GPU entry
            gpus.push(GpuInfo {
                gpu_id: format!("{}-gpu0", backend.id),
                gpu_name: "GPU0".to_string(),
                device_index: 0,
                layers_loaded: HashMap::new(),
                free_memory: backend.vram_capacity.saturating_sub(total_vram_used),
                total_memory: backend.vram_capacity,
            });
            
            debug!("{}: Inferred GPU usage: {:.2}GB used / {:.2}GB total", 
                backend.name,
                total_vram_used as f64 / (1024.0 * 1024.0 * 1024.0),
                backend.vram_capacity as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
        return;
    }
    
    // Calculate total VRAM used from all models
    let mut total_vram_used: u64 = 0;
    let mut model_count = 0;
    let mut found_models = Vec::new();
    
    for model in models {
        if let Some(size_vram) = model.get("size_vram").and_then(|s| s.as_u64()) {
            total_vram_used += size_vram;
            model_count += 1;
            
            // Extract model name for tracking
            if let Some(name) = model.get("name").and_then(|n| n.as_str()) {
                found_models.push(name.to_string());
                
                // Update layers_loaded if we can infer it
                // For now, just track that the model is loaded
                if let Some(gpu) = gpus.first_mut() {
                    gpu.layers_loaded.insert(name.to_string(), 1); // Placeholder
                }
            }
        }
    }
    
    // Populate loaded_models list for affinity routing
    {
        let mut loaded_lock = backend.loaded_models.write().await;
        *loaded_lock = found_models;
    }
    
    // Update free memory on primary GPU
    if let Some(primary_gpu) = gpus.first_mut() {
        let old_free = primary_gpu.free_memory;
        primary_gpu.free_memory = primary_gpu.total_memory.saturating_sub(total_vram_used);
        
        let free_gb = primary_gpu.free_memory as f64 / (1024.0 * 1024.0 * 1024.0);
        let used_gb = total_vram_used as f64 / (1024.0 * 1024.0 * 1024.0);
        
        // Only log if there's a significant change (>100MB)
        if old_free.abs_diff(primary_gpu.free_memory) > 100 * 1024 * 1024 {
            info!("🔄 {}: VRAM updated - {} models, {:.2}GB used, {:.2}GB free",
                backend.name, model_count, used_gb, free_gb);
        }
    }
}

async fn clear_gpu_usage(backend: &Backend) {
    let mut gpus = backend.gpus.write().await;
    
    for gpu in gpus.iter_mut() {
        let old_free = gpu.free_memory;
        gpu.free_memory = gpu.total_memory;
        gpu.layers_loaded.clear();
        
        // Log if there was a significant change
        if old_free.abs_diff(gpu.free_memory) > 100 * 1024 * 1024 {
            info!("✨ {}: All models unloaded, {:.2}GB freed", 
                backend.name,
                (gpu.free_memory - old_free) as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
    }
}

/// Helper: Get current VRAM usage for a backend (for external use)
#[allow(dead_code)]
pub async fn get_backend_vram_usage(backend: &Backend) -> (u64, u64) {
    let gpus = backend.gpus.read().await;
    
    if let Some(primary_gpu) = gpus.first() {
        let used = primary_gpu.total_memory.saturating_sub(primary_gpu.free_memory);
        (used, primary_gpu.total_memory)
    } else {
        (0, backend.vram_capacity)
    }
}
