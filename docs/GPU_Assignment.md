# 🎯 Primary GPU Assignment

This document explains how to configure primary GPU assignments for containers in the Ollama Load Balancer.

## Overview

The load balancer supports running multiple containers where each container:
- Has access to **all GPUs** in the system
- Uses a **different GPU as primary** (GPU 0 from container's perspective)
- Can fall back to other GPUs if needed

This allows optimal load distribution across your GPU pool.

## How It Works

### GPU Visibility Environment Variables

**For NVIDIA GPUs:**
- `NVIDIA_VISIBLE_DEVICES=all` → Makes all GPUs visible to the container
- `CUDA_VISIBLE_DEVICES=X,Y,Z` → Reorders GPUs so GPU X becomes primary (GPU 0)

**For AMD GPUs:**
- `ROCR_VISIBLE_DEVICES=X,Y,Z` → Reorders GPUs so GPU X becomes primary
- `HIP_VISIBLE_DEVICES=X,Y,Z` → Same reordering for HIP runtime

### Example

If you have 3 NVIDIA GPUs and set `CUDA_VISIBLE_DEVICES=1,2,0`:
- Physical GPU 1 → Container sees as GPU 0 (primary)
- Physical GPU 2 → Container sees as GPU 1
- Physical GPU 0 → Container sees as GPU 2

## Configuration Methods

### Method 1: Dynamic Generation (Recommended)

Use the `generate-compose.sh` script to automatically configure everything:

```bash
# 1. Edit .env and set your desired container counts
vim .env
# Example: OLLAMA_NVIDIA_REPLICAS=3, OLLAMA_AMD_REPLICAS=2

# 2. Generate configuration files
./generate-compose.sh

# 3. Start services
docker compose up -d
```

**What the script does:**
1. Reads `OLLAMA_NVIDIA_REPLICAS` and `OLLAMA_AMD_REPLICAS` from `.env`.
2. Reads `TOTAL_NVIDIA_GPUS` and `TOTAL_AMD_GPUS` (defaults: 3, 2).
3. Auto-generates GPU rotation assignments in `.env`:
   - Uses modulo arithmetic against Total GPUs to assign primaries.
   - Example (1 NVIDIA Replica, 3 GPUs): `NVIDIA_GPU_1=0,1,2` (Uses *all* GPUs).
4. Creates `docker-compose.yml` with individual container definitions.

### Method 2: Manual Configuration

Edit `.env` to customize GPU assignments:

```bash
# Set container counts
OLLAMA_NVIDIA_REPLICAS=3
OLLAMA_AMD_REPLICAS=2

# Customize GPU assignments (these override auto-rotation)
NVIDIA_GPU_1=0,1,2  # Container 1 uses GPU 0 as primary
NVIDIA_GPU_2=1,2,0  # Container 2 uses GPU 1 as primary
NVIDIA_GPU_3=2,0,1  # Container 3 uses GPU 2 as primary

AMD_GPU_1=0,1
AMD_GPU_2=0,1  # NOTE: Avoid 1,0 order on some AMD drivers as it may fail initialization
```

> **Warning (AMD Users)**: Some AMD configurations fail to initialize if the device list is non-monotonic (e.g., `1,0`). If you encounter `gpu_count=0` errors or CPU fallback, use sequential ordering (e.g., `0,1`) for all containers.

Then run the generator:
```bash
./generate-compose.sh
docker compose up -d
```

## Verification

### Check GPU Assignments Inside Containers

**For NVIDIA containers:**
```bash
docker exec ollama-nvidia-1 nvidia-smi
# GPU 0 should be physical GPU 0

docker exec ollama-nvidia-2 nvidia-smi
# GPU 0 should be physical GPU 1

docker exec ollama-nvidia-3 nvidia-smi
# GPU 0 should be physical GPU 2
```

**For AMD containers:**
```bash
docker exec ollama-amd-1 rocm-smi

docker exec ollama-amd-2 rocm-smi
```

### Monitor GPU Usage
```bash
# On host
nvidia-smi  # or rocm-smi for AMD

# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

## Use Cases

### Scenario 1: Equal Distribution (3 GPUs, 3 Containers)
```bash
OLLAMA_NVIDIA_REPLICAS=3
# Auto-generated assignments:
# NVIDIA_GPU_1=0,1,2
# NVIDIA_GPU_2=1,2,0
# NVIDIA_GPU_3=2,0,1
```
Each container primarily uses a different GPU.

### Scenario 2: Oversubscription (2 GPUs, 4 Containers)
```bash
OLLAMA_AMD_REPLICAS=4
# Auto-generated assignments:
# AMD_GPU_1=0,1
# AMD_GPU_2=1,0
# AMD_GPU_3=0,1
# AMD_GPU_4=1,0
```
Containers share GPUs with rotating primaries.

### Scenario 3: Custom Priority
```bash
# Give container 1 priority to newest GPU
NVIDIA_GPU_1=2,1,0
NVIDIA_GPU_2=1,2,0
NVIDIA_GPU_3=0,1,2
```

## Troubleshooting

### Container not using expected GPU
1. Check environment variables:
   ```bash
   docker inspect ollama-nvidia-1 | grep CUDA_VISIBLE_DEVICES
   ```

2. Verify GPU is visible:
   ```bash
   docker exec ollama-nvidia-1 nvidia-smi -L
   ```

### All containers using same GPU
- The `.env` GPU assignment variables may not be generated
- Run `./generate-compose.sh` to regenerate configuration
- Check that variables like `NVIDIA_GPU_1`, `NVIDIA_GPU_2`, etc. exist in `.env`

### Performance not improving with multiple containers
- Ensure your models are small enough to fit in VRAM with parallel execution
- Check OLLAMA_NUM_PARALLEL settings
- Monitor GPU memory usage with `nvidia-smi` or `rocm-smi`

## Advanced: Manual docker-compose.yml Editing

If you prefer to edit `docker-compose.yml` directly instead of using the generator:

```yaml
services:
  ollama-nvidia-1:
    image: ollama/ollama:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1,2  # GPU 0 primary
    # ... rest of config

  ollama-nvidia-2:
    image: ollama/ollama:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=1,2,0  # GPU 1 primary
    # ... rest of config
```

## See Also

- [README.md](../README.md) - Main documentation
- [generate-compose.sh](../generate-compose.sh) - Dynamic configuration script
- [.env.example](../.env.example) - Configuration template
