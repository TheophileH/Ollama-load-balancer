# ⚖️ Ollama Load Balancer

A high-performance, unified Load Balancer for Ollama that seamlessly routes requests between **NVIDIA** and **AMD** GPUs. Compatible with all standard Ollama clients.


## 🛠️ Architecture

```mermaid
flowchart TD
    User([User / Client]) -->|Port 11434| LB[Load Balancer]
    
    subgraph Docker Internal Network
        subgraph NvidiaCluster [NVIDIA Scalable Backend]
            Nvidia1["Ollama NVIDIA #1"]
            Nvidia2["Ollama NVIDIA #2"]
            NvidiaN["Ollama NVIDIA #n"]
        end
        
        subgraph AmdCluster [AMD Scalable Backend]
            Amd1["Ollama AMD #1"]
            Amd2["Ollama AMD #2"]
            AmdN["Ollama AMD #n"]
        end

        LB -->|Discovers & Routes| Nvidia1
        LB -->|Discovers & Routes| Nvidia2
        LB -->|Discovers & Routes| NvidiaN
        LB -->|Discovers & Routes| Amd1
        LB -->|Discovers & Routes| Amd2
        LB -->|Discovers & Routes| AmdN
    end
    
    subgraph Hardware
        Nvidia1 & Nvidia2 & NvidiaN -->|CUDA| GPU_N["NVIDIA GPUs"]
        Amd1 & Amd2 & AmdN -->|ROCm| GPU_A["AMD GPUs"]
    end
```

The system consists of three main service types:
1.  **`load-balancer`**: The Rust (or Python) proxy listening on 11434.
2.  **`ollama-nvidia`**: Scalable CUDA backend (Default: 1 replica).
3.  **`ollama-amd`**: Scalable ROCm backend (Default: 1 replica).

> **Note**: All backend instances share a single **Shared Storage Volume** (`ollama`) to ensure models are available across all replicas without duplication.

---

## 🚀 Features

*   **Unified Interface**: Exposes a standard Ollama API on port `11434`. Compatible with all standard Ollama clients (CLI, Web UIs, Libraries).
*   **Dual-Backend Support**:
    *   **NVIDIA**: Runs standard CUDA-accelerated Ollama.
    *   **AMD**: Runs ROCm-accelerated Ollama for Radeon/Instinct cards.
*   **Dynamic Scaling & Discovery**:
    *   Scale `nvidia` or `amd` backends to multiple replicas (e.g., 3 NVIDIA nodes).
    *   Load Balancer detects new instances automatically via Docker Socket.
    *   Traffic is distributed across all available replicas of the target type.
*   **Smart Aggregation**: Merges model lists (`/api/tags`) and running process lists (`/api/ps`) from both instances into a single view.
*   **Shared Storage**: Both instances use a single shared volume for models, ensuring consistency and preventing duplication.
*   **Auto-Detection of VRAM**: Automatically inspects containers via `nvidia-smi` or `rocm-smi` to determine available VRAM if not manually configured.
*   **Manual Selection**: Force specific requests to a specific backend using:
    *   **Prefix**: `amd/tinyllama`
    *   **Suffix**: `tinyllama@amd`
*   **Load Balancing Strategies**:
    *   **Model Affinity** (Best for Performance): Routes to instance with model already loaded. If not found, falls back to `VRAM` or `Round Robin`.
    *   **Least Connection**: Routes to least busy instance (used as secondary strategy).
    *   **VRAM**: Routes to instance with most free memory.
    *   **Round Robin**: Distributes requests evenly.
    *   **Preference**: Fixed routing to `nvidia` or `amd`.

---

## 🏗️ Implementations

This project offers two interchangeable implementations:

### 1. 🦀 Rust (Default)
Located in `Rust_lb/`.
*   **Performance**: Native machine code, extremely low latency, tiny memory footprint (<10MB).
*   **Features**: Full support for Dynamic Scaling, Docker Discovery, and Auto-VRAM.
*   **Recommended for**: Production.

### 2. 🐍 Python
Located in `Python_lb/`.
*   **Stack**: FastAPI, Uvicorn.
*   **Features**: Includes feature-parity with Rust (inc. Dynamic Scaling, Auto-VRAM).
*   **Recommended for**: Easy modification/prototyping.

---

## ⚡ Quick Start

### 1. Configure Environment
Copy the example configuration:
```bash
cp .env.example .env
```

**Choose your Implementation** in `.env`:
```ini
# Default: Rust
LB_BUILD_CONTEXT=./Rust_lb
LB_DOCKERFILE=Dockerfile

# To switch to Python, uncomment:
# LB_BUILD_CONTEXT=./Python_lb
# LB_DOCKERFILE=Dockerfile
```

### 2. Start Services
```bash
docker compose up -d --build
```

### 3. Verify
```bash
curl http://localhost:11434/api/version
```

---

## 📖 Usage

### Standard Commands
The Load Balancer makes your dual-GPU (or multi-node) setup look like a single Ollama instance.
```bash
# Run on default/vram-available backend
ollama run llama3
```

### Dynamic Scaling
Scale your backends on the fly. The Load Balancer discovers them instantly.
```bash
# Run 3 NVIDIA instances and 2 AMD instances
docker compose up -d --scale ollama-nvidia=3 --scale ollama-amd=2
```

### Single-Vendor Mode
If you only have one type of GPU (e.g., only NVIDIA), you can disable the other vendor by setting its replicas to `0` in `.env` (or via CLI).
```bash
# Run only NVIDIA instances (AMD disabled)
docker compose up -d --scale ollama-amd=0 --scale ollama-nvidia=1
```

### 🖥️  CLI Tool (olb)

The project includes a powerful Rust-based CLI Tool (`olb`) for enhanced management and interactive model discovery.

**Highlights**:
*   **Interactive Discovery**: Browse `ollama.com` models with a TUI.
*   **Smart Selection**: Jump to items by typing their number index (e.g., `1`), or fuzzy search by name.
*   **Detailed Metadata**: View context window sizes, input types, and more before downloading.

-> **[📖 Read the Full CLI Documentation](docs/CLI.md)**

**Quick Install**:
```bash
cd Rust_lb && cargo build --release && cp target/release/olb ~/.local/bin/olb
```

**Usage**:
```bash
# Search for models (interactive)
olb discover "llama3"

# Pull a model
olb pull "llama3"
```

### Seamless Integration (Recommended)
To use `discover` seamlessly as `ollama discover`, add this wrapper to your shell configuration (`~/.bashrc` or `~/.zshrc`).
**Note**: This assumes `olb` is in your `$PATH` (e.g., installed to `~/.local/bin`).

```bash
echo '
ollama() {
    if [ "$1" = "discover" ]; then
        olb discover "${@:2}"
    else
        command ollama "$@"
    fi
}' >> ~/.bashrc
source ~/.bashrc
```

**Verify the setup**:
```bash
ollama discover llama
```
<img width="1120" height="616" alt="image" src="https://github.com/user-attachments/assets/79585a46-6111-46e1-ae50-9a3a885fd66e" />

<img width="1718" height="616" alt="image" src="https://github.com/user-attachments/assets/e53bb227-8838-409f-9546-8adff7a34950" />


### Manual Instance Selection
Force execution on a specific GPU vendor:

**Method 1: Prefixes (Clean, recommended for CLI)**
```bash
ollama run amd/tinyllama
ollama run nvidia/llama3
```

**Method 2: Suffixes**
```bash
ollama run tinyllama@amd
ollama run llama3@nvidia
```

### Advanced Routing & Prioritization
Control strategy via `.env` or Headers.

**Strategies available in `.env`**:
*   `vram` (Default): Use backend with most free memory.
*   `round_robin`: Random distribution.
*   `nvidia` / `amd`: Strict preference.

**Request Overrides (Headers)**:
```bash
curl http://localhost:11434/api/generate \
  -d '{"model": "llama3.1", "prompt": "Why is the sky blue?"}' \
  -H "X-Load-Balancing-Strategy: amd" \
  -H "X-Enforce-Strategy: true"
```

---

## ⚙️ Configuration

Centralized in `.env`.

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Load Balancer external port | `11434` |
| `LB_BUILD_CONTEXT` | Path to source code (`./Rust_lb` or `./Python_lb`) | `./Rust_lb` |
| `LOAD_BALANCING_STRATEGY`| Routing logic (`vram`, `round_robin`, `nvidia`, `amd`) | `vram` |
| `OLLAMA_ENABLE_AFFINITY` | Enable Sticky Routing & Least Connection (`true`/`false`) | `false` |
| `OLLAMA_NVIDIA_REPLICAS`| Default number of NVIDIA containers | `1` |
| `OLLAMA_AMD_REPLICAS`| Default number of AMD containers | `1` |
| `OLLAMA_NVIDIA_VRAM_MB` | Total VRAM per NVIDIA instance (MB). Leave empty to Autodetect. | *Autodetect* |
| `OLLAMA_AMD_VRAM_MB` | Total VRAM per AMD instance (MB). Leave empty to Autodetect. | *Autodetect* |
| `LOG_LEVEL` | Log verbosity (`info`, `debug`, `trace`, `warn`, `error`) | `info` |

---

## 🔧 Troubleshooting

*   **"Error: something went wrong" (CLI)**:
    *   Check `docker compose logs load-balancer`.
    *   Ensure you are using the latest version of this project which fixes the CLI payload handling bug.
*   **Streaming timeouts**:
    *   Verify your request client allows keeping connections open. The LB handles backpressure correctly.
*   **AMD GPU not detected**:
    *   Check the host's ROCm version.
    *   Ensure `ROCR_VISIBLE_DEVICES` in `.env` matches your hardware ID.

---

## 📝 Roadmap / TODO

*   **Debug `/api/ps` for AMD**: Investigate why `ollama ps` does not show models currently served on AMD backend instances.

---

## 🤝 Contributing
Contributions are welcome!

## 🤝 Author

**[TheophileH](https://github.com/TheophileH)**

## 📄 License
MIT License.
