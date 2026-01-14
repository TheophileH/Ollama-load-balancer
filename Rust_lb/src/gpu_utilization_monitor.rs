use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, debug};
use crate::discovery::BackendType;

#[derive(Clone, Debug)]
pub struct GpuUtilization {
    pub gpu_index: usize,
    pub backend_type: BackendType,
    pub utilization_percent: u8,  // 0-100
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
}

pub type GpuUtilState = Arc<RwLock<Vec<GpuUtilization>>>;

/// Poll NVIDIA GPUs using nvidia-smi on the HOST (not in container)
async fn poll_nvidia_gpus() -> Result<Vec<GpuUtilization>, String> {
    // Execute nvidia-smi on the HOST using nsenter to access host namespace
    // nsenter --target 1 --mount --uts --ipc --net --pid runs command in host PID 1 namespace
    match Command::new("nsenter")
        .args(&[
            "--target", "1",
            "--mount", "--uts", "--ipc", "--net", "--pid",
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ])
        .output()
    {
        Ok(output) => {
            if !output.status.success() {
                return Err(format!("nvidia-smi failed: {}", String::from_utf8_lossy(&output.stderr)));
            }
            
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut gpus = Vec::new();
            
            for line in stdout.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 4 {
                    if let (Ok(idx), Ok(util), Ok(used), Ok(total)) = (
                        parts[0].parse::<usize>(),
                        parts[1].parse::<u8>(),
                        parts[2].parse::<u64>(),
                        parts[3].parse::<u64>(),
                    ) {
                        gpus.push(GpuUtilization {
                            gpu_index: idx,
                            backend_type: BackendType::Nvidia,
                            utilization_percent: util,
                            memory_used_mb: used,
                            memory_total_mb: total,
                        });
                    }
                }
            }
            
            Ok(gpus)
        }
        Err(e) => Err(format!("Failed to execute nvidia-smi: {}", e)),
    }
}

/// Poll AMD GPUs using rocm-smi on the HOST (not in container)
async fn poll_amd_gpus() -> Result<Vec<GpuUtilization>, String> {
    // Execute rocm-smi on the HOST using nsenter to get memory usage
    match Command::new("nsenter")
        .args(&[
            "--target", "1",
            "--mount", "--uts", "--ipc", "--net", "--pid",
            "rocm-smi",
            "--showmeminfo", "vram",
            "--csv"
        ])
        .output()
    {
        Ok(output) => {
            if !output.status.success() {
                return Err(format!("rocm-smi failed: {}", String::from_utf8_lossy(&output.stderr)));
            }
            
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut gpus = Vec::new();
            
            // Parse CSV output from rocm-smi
            // Expected format: device,VRAM Total Memory (B),VRAM Total Used Memory (B)
            // Example: card0,8573157376,16490496
            for (idx, line) in stdout.lines().enumerate() {
                if idx == 0 { continue; } // Skip header
                
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    // Extract GPU index from device name (e.g., "card0" -> 0, "card1" -> 1)
                    let gpu_idx = if parts[0].starts_with("card") {
                       parts[0].trim_start_matches("card").parse::<usize>().unwrap_or(idx - 1)
                    } else {
                        // Fallback to line index if format doesn't match
                        idx - 1
                    };
                    
                    if let (Ok(total_bytes), Ok(used_bytes)) = (
                        parts[1].parse::<u64>(),
                        parts[2].parse::<u64>(),
                    ) {
                        let used_mb = used_bytes / (1024 * 1024);
                        let total_mb = total_bytes / (1024 * 1024);
                        
                        // Debug: Log the parsed values
                        info!("🔍 AMD GPU {}: parsed total_bytes={}, used_bytes={} → {}MB / {}MB", 
                            gpu_idx, total_bytes, used_bytes, used_mb, total_mb);
                        
                        // Get utilization percentage (separate command)
                        let util_percent = get_amd_gpu_utilization(gpu_idx).await.unwrap_or(0);
                        
                        gpus.push(GpuUtilization {
                            gpu_index: gpu_idx,
                            backend_type: BackendType::Amd,
                            utilization_percent: util_percent,
                            memory_used_mb: used_mb,
                            memory_total_mb: total_mb,
                        });
                    }
                }
            }
            
            Ok(gpus)
        }
        Err(e) => Err(format!("Failed to execute rocm-smi: {}", e)),
    }
}

/// Get AMD GPU utilization percentage
async fn get_amd_gpu_utilization(gpu_idx: usize) -> Result<u8, String> {
    match Command::new("nsenter")
        .args(&[
            "--target", "1",
            "--mount", "--uts", "--ipc", "--net", "--pid",
            "rocm-smi",
            "--showuse",
            "--csv"
        ])
        .output()
    {
        Ok(output) => {
            if !output.status.success() {
                return Ok(0);
            }
            
            let stdout = String::from_utf8_lossy(&output.stdout);
            for (idx, line) in stdout.lines().enumerate() {
                if idx == 0 { continue; } // Skip header
                
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 2 {
                    if let (Ok(parsed_idx), Ok(util)) = (
                        parts[0].parse::<usize>(),
                        parts[1].trim_end_matches('%').parse::<u8>(),
                    ) {
                        if parsed_idx == gpu_idx {
                            return Ok(util);
                        }
                    }
                }
            }
            Ok(0)
        }
        Err(_) => Ok(0),
    }
}

/// Main GPU monitoring loop - polls every 1 second
pub async fn start_gpu_monitor(gpu_state: GpuUtilState) {
    info!("🔬 Starting GPU utilization monitor (polling every 1s)");
    
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    
    loop {
        interval.tick().await;
        
        let mut all_gpus = Vec::new();
        
        // Poll NVIDIA GPUs
        match poll_nvidia_gpus().await {
            Ok(nvidia_gpus) => {
                if !nvidia_gpus.is_empty() {
                    info!("📊 Polled {} NVIDIA GPUs", nvidia_gpus.len());
                    for gpu in &nvidia_gpus {
                        debug!("  GPU {}: {}% util, {}MB / {}MB VRAM", 
                            gpu.gpu_index, gpu.utilization_percent, 
                            gpu.memory_used_mb, gpu.memory_total_mb);
                    }
                    all_gpus.extend(nvidia_gpus);
                }
            }
            Err(e) => {
                debug!("NVIDIA polling failed (expected if no NVIDIA GPUs): {}", e);
            }
        }
        
        // Poll AMD GPUs (always try, even if NVIDIA succeeded - for hybrid systems)
        match poll_amd_gpus().await {
            Ok(amd_gpus) => {
                if !amd_gpus.is_empty() {
                    info!("📊 Polled {} AMD GPUs", amd_gpus.len());
                    for gpu in &amd_gpus {
                        debug!("  GPU {}: {}% util, {}MB / {}MB VRAM", 
                            gpu.gpu_index, gpu.utilization_percent, 
                            gpu.memory_used_mb, gpu.memory_total_mb);
                    }
                    all_gpus.extend(amd_gpus);
                }
            }
            Err(e) => {
                debug!("AMD polling failed (expected if no AMD GPUs): {}", e);
            }
        }
        
        // Update state with combined GPU list
        if !all_gpus.is_empty() {
            *gpu_state.write().await = all_gpus;
        }
    }
}

/// Get utilization for a specific GPU index
#[allow(dead_code)]
pub async fn get_gpu_utilization(gpu_state: &GpuUtilState, gpu_index: usize) -> Option<GpuUtilization> {
    let state = gpu_state.read().await;
    state.iter().find(|g| g.gpu_index == gpu_index).cloned()
}
