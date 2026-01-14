#!/bin/bash
# Dynamic Docker Compose Generator for Ollama Load Balancer with GPU Assignment
# This script generates docker-compose.yml based on the number of GPUs specified in .env

set -e

# Load environment variables from .env
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found!"
    exit 1
fi

# Default values
NVIDIA_REPLICAS=${OLLAMA_NVIDIA_REPLICAS:-0}
AMD_REPLICAS=${OLLAMA_AMD_REPLICAS:-0}

# Total Physical GPUs (Defaults to previously known hardware config)
# You can override these in .env if you add OLLAMA_NVIDIA_GPUS=x
TOTAL_NVIDIA_GPUS=${OLLAMA_NVIDIA_GPUS:-3}
TOTAL_AMD_GPUS=${OLLAMA_AMD_GPUS:-2}

echo "=== Ollama Load Balancer - Docker Compose Generator ==="
echo ""
echo "Configuration:"
echo "  NVIDIA containers: $NVIDIA_REPLICAS"
echo "  AMD containers: $AMD_REPLICAS"
echo ""

# Function to generate GPU rotation for a given container index and total GPUs
generate_gpu_rotation() {
    local container_idx=$1
    local total_gpus=$2
    local gpu_order=""
    
    for j in $(seq 0 $((total_gpus - 1))); do
        gpu_id=$(( (container_idx - 1 + j) % total_gpus ))
        if [ -n "$gpu_order" ]; then
            gpu_order="${gpu_order},${gpu_id}"
        else
            gpu_order="${gpu_id}"
        fi
    done
    
    echo "$gpu_order"
}

# Update .env file with GPU assignments
echo "Step 1: Updating .env with GPU assignments..."

# Create a temporary file
TMP_ENV=$(mktemp)

# Read .env and update/add GPU assignment section
IN_GPU_SECTION=false
SKIP_LINE=false

while IFS= read -r line || [ -n "$line" ]; do
    # Detect start of GPU assignment section
    if [[ "$line" =~ ^#\ GPU\ Assignment ]]; then
        IN_GPU_SECTION=true
        echo "# GPU Assignment (Primary GPU for each container) - AUTO-GENERATED" >> "$TMP_ENV"
        echo "# Format: \"primary,secondary,tertiary\" - first GPU in list becomes GPU 0 in container" >> "$TMP_ENV"
        
        # Generate NVIDIA GPU assignments
        if [ $NVIDIA_REPLICAS -gt 0 ]; then
            echo "# NVIDIA GPU Assignment" >> "$TMP_ENV"
            for i in $(seq 1 $NVIDIA_REPLICAS); do
                rotation=$(generate_gpu_rotation $i $TOTAL_NVIDIA_GPUS)
                echo "NVIDIA_GPU_$i=$rotation" >> "$TMP_ENV"
            done
        fi
        
        # Generate AMD GPU assignments
        if [ $AMD_REPLICAS -gt 0 ]; then
            echo "# AMD GPU Assignment" >> "$TMP_ENV"
            for i in $(seq 1 $AMD_REPLICAS); do
                rotation=$(generate_gpu_rotation $i $TOTAL_AMD_GPUS)
                echo "AMD_GPU_$i=$rotation" >> "$TMP_ENV"
            done
        fi
        
        continue
    fi
    
    # Skip old GPU assignment lines
    if [ "$IN_GPU_SECTION" = true ]; then
        if [[ "$line" =~ ^(NVIDIA_GPU_|AMD_GPU_|#\ NVIDIA|#\ AMD|#\ Format) ]]; then
            continue
        elif [[ "$line" =~ ^$ ]]; then
            # Empty line might signal end of section, but keep going
            echo "$line" >> "$TMP_ENV"
            continue
        elif [[ "$line" =~ ^# ]] && [[ ! "$line" =~ ^#\ (NVIDIA|AMD|Format) ]]; then
            # New comment section, exit GPU section
            IN_GPU_SECTION=false
            echo "$line" >> "$TMP_ENV"
        elif [[ ! "$line" =~ ^# ]]; then
            # Non-comment, non-GPU line, exit GPU section
            IN_GPU_SECTION=false
            echo "$line" >> "$TMP_ENV"
        else
            echo "$line" >> "$TMP_ENV"
        fi
    else
        echo "$line" >> "$TMP_ENV"
    fi
done < .env

# Replace .env with updated version
mv "$TMP_ENV" .env
echo "  ✓ Updated .env with GPU assignments"
echo ""

echo "Step 2: Generating docker-compose.yml..."

# Start writing docker-compose.yml
cat > docker-compose.yml << 'EOF'
services:
  # --- Load Balancer (Rust Version) ---
  load-balancer:
    build:
      context: ${LB_BUILD_CONTEXT:-./Rust_lb}
      dockerfile: ${LB_DOCKERFILE:-Dockerfile}
    container_name: ${OLLAMA_LB_CONTAINER_NAME:-load-balancer}
    pid: "host" # Required for nsenter
    privileged: true # Required for nsenter
    ports:
      - "${PORT:-11434}:11434"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    environment:
      # Map generic LOG_LEVEL to Rust's specific env var
      - RUST_LOG=${LOG_LEVEL:-info}
      # Also pass LOG_LEVEL for Python fallback
      - LOG_LEVEL=${LOG_LEVEL:-info}
    env_file: .env

    deploy:
      resources:
        limits:
          memory: 128M # Rust is tiny
    restart: always
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:11434/api/version" ]
      interval: 30s
      timeout: 10s
      retries: 3

EOF

# Generate NVIDIA containers
for i in $(seq 1 $NVIDIA_REPLICAS); do
    # Calculate GPU order (rotate GPUs)
    # Container 1: 0,1,2  Container 2: 1,2,0  Container 3: 2,0,1
    total_gpus=$TOTAL_NVIDIA_GPUS
    gpu_order=""
    for j in $(seq 0 $((total_gpus - 1))); do
        gpu_id=$(( (i - 1 + j) % total_gpus ))
        if [ -n "$gpu_order" ]; then
            gpu_order="${gpu_order},${gpu_id}"
        else
            gpu_order="${gpu_id}"
        fi
    done
    
    # Use custom GPU assignment if defined, otherwise use calculated rotation
    gpu_var="NVIDIA_GPU_${i}"
    default_gpu="${gpu_order}"
    
    cat >> docker-compose.yml << EOF
  # --- NVIDIA Backend Container $i (Primary GPU $((i-1))) ---
  ollama-nvidia-$i:
    image: \${OLLAMA_NVIDIA_IMAGE:-ollama/ollama:latest}
    container_name: ollama-nvidia-$i
    runtime: nvidia
    labels:
      - "ollama.backend=true"
      - "ollama.type=nvidia"
    ports:
      - "11434"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=\${${gpu_var}:-${default_gpu}}
      - OLLAMA_HOST=\${OLLAMA_HOST:-0.0.0.0:11434}
      - OLLAMA_ORIGINS=\${OLLAMA_ORIGINS:-*}
      - OLLAMA_MODELS=/root/.ollama/models
      - OLLAMA_NUM_PARALLEL=\${OLLAMA_NUM_PARALLEL:-4}
      - OLLAMA_FLASH_ATTENTION=\${OLLAMA_FLASH_ATTENTION:-1}
      - OLLAMA_KV_CACHE_TYPE=\${OLLAMA_KV_CACHE_TYPE:-f16}
      - OLLAMA_CONTEXT_LENGTH=\${OLLAMA_CONTEXT_LENGTH:-2048}
    volumes:
      - ollama_storage:/root/.ollama
    restart: always

EOF
done

# Generate AMD containers
for i in $(seq 1 $AMD_REPLICAS); do
    # Calculate GPU order (rotate GPUs)
    total_gpus=$TOTAL_AMD_GPUS
    gpu_order=""
    for j in $(seq 0 $((total_gpus - 1))); do
        gpu_id=$(( (i - 1 + j) % total_gpus ))
        if [ -n "$gpu_order" ]; then
            gpu_order="${gpu_order},${gpu_id}"
        else
            gpu_order="${gpu_id}"
        fi
    done
    
    # Use custom GPU assignment if defined, otherwise use calculated rotation
    gpu_var="AMD_GPU_${i}"
    default_gpu="${gpu_order}"
    
    cat >> docker-compose.yml << EOF
  # --- AMD Backend Container $i (Primary GPU $((i-1))) ---
  ollama-amd-$i:
    image: \${OLLAMA_AMD_IMAGE:-ollama/ollama:rocm}
    container_name: ollama-amd-$i
    labels:
      - "ollama.backend=true"
      - "ollama.type=amd"
    ports:
      - "11434"
    devices:
      - /dev/kfd
      - /dev/dri
    environment:
      - OLLAMA_HOST=\${OLLAMA_HOST:-0.0.0.0:11434}
      - OLLAMA_ORIGINS=\${OLLAMA_ORIGINS:-*}
      - OLLAMA_MODELS=/root/.ollama/models
      - OLLAMA_NUM_PARALLEL=\${OLLAMA_NUM_PARALLEL:-4}
      # AMD specific envs
      - HSA_OVERRIDE_GFX_VERSION=\${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
      - HCC_AMDGPU_TARGET=\${HCC_AMDGPU_TARGET:-gfx1030}
      - ROCR_VISIBLE_DEVICES=\${${gpu_var}:-${default_gpu}}
      - HIP_VISIBLE_DEVICES=\${${gpu_var}:-${default_gpu}}
      - OLLAMA_CONTEXT_LENGTH=\${OLLAMA_CONTEXT_LENGTH:-2048}
      - OLLAMA_DEBUG=\${OLLAMA_DEBUG}
    group_add:
      - "44"
      - "109"
    security_opt:
      - seccomp:unconfined
    volumes:
      - ollama_storage:/root/.ollama
    restart: always

EOF
done

cat >> docker-compose.yml << 'EOF'
volumes:
  ollama_storage:
    name: ${OLLAMA_DATA_VOLUME:-ollama}

EOF

echo "  ✓ docker-compose.yml generated successfully!"
echo ""
echo "=== Generation Complete ==="
echo ""
echo "Summary:"
echo "  - Load balancer: 1 container"
echo "  - NVIDIA containers: $NVIDIA_REPLICAS"
echo "  - AMD containers: $AMD_REPLICAS"
echo ""
echo "Files updated:"
echo "  ✓ .env (with GPU assignments)"
echo "  ✓ docker-compose.yml"
echo ""
echo "GPU Assignments in .env:"
if [ $NVIDIA_REPLICAS -gt 0 ]; then
    for i in $(seq 1 $NVIDIA_REPLICAS); do
        rotation=$(generate_gpu_rotation $i $TOTAL_NVIDIA_GPUS)
        echo "  NVIDIA_GPU_$i=$rotation"
    done
fi
if [ $AMD_REPLICAS -gt 0 ]; then
    for i in $(seq 1 $AMD_REPLICAS); do
        rotation=$(generate_gpu_rotation $i $TOTAL_AMD_GPUS)
        echo "  AMD_GPU_$i=$rotation"
    done
fi
echo ""
echo "You can manually edit GPU assignments in .env if needed."
echo "Then run: docker compose up -d"

