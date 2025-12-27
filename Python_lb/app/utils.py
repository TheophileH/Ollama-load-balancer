import docker

def autodetect_vram(container_id: str, b_type: str) -> int:
    """Attempt to detect VRAM size (in bytes) by executing commands inside the container."""
    try:
        client = docker.from_env()
        container = client.containers.get(container_id)
        
        if b_type == "nvidia":
            # Nvidia SMI
            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
            res = container.exec_run(cmd)
            if res.exit_code == 0:
                # Output might be multi-line (multiple GPUs). Sum them.
                total_mb = 0
                for line in res.output.decode().splitlines():
                    if line.strip().isdigit():
                        total_mb += int(line.strip())
                if total_mb > 0:
                    return total_mb * 1024 * 1024

        elif b_type == "amd":
            # ROCm SMI (Try standard path or PATH)
            # JSON format is easiest: --showvram --json
            cmd = "rocm-smi --showvram --json"
            res = container.exec_run(cmd)
            if res.exit_code == 0:
                import json
                try:
                    data = json.loads(res.output.decode())
                    total_bytes = 0
                    for card in data:
                        # Format varies, sometimes "card0" -> {"VRAM Total Memory (B)": ...}
                        # or list. Verify structure.
                        # Fallback to text parsing if JSON fails?
                        # Let's try parsing the values recursively for keys like "VRAM Total Memory (B)"
                        if isinstance(data[card], dict):
                             val = data[card].get("VRAM Total Memory (B)")
                             if val: total_bytes += int(val)
                    if total_bytes > 0:
                        return total_bytes
                except:
                    pass
            
            # Fallback for AMD: sysfs
            # We assume card0, card1... mapped? 
            # Docker container usually sees /dev/dri/cardX. 
            # Sysfs might not be mounted unless -v /sys is passed? 
            # Usually not. relying on rocm-smi is safer.
            pass

    except Exception as e:
        print(f"Failed to autodetect VRAM for {container_id}: {e}")
    
    return 0

# Cache capacity to avoid repeated execs
_capacity_cache = {}

async def get_vram_stats():
    """Calculate free VRAM for each backend instance."""
    stats = {}
    
    backends = registry.get_instances()
    if not backends:
        return stats

    for b in backends:
        # 1. Configured Limit
        capacity_mb = int(os.getenv(f"OLLAMA_{b.type.upper()}_VRAM_MB", "0"))
        capacity = capacity_mb * 1024 * 1024
        
        # 2. Autodetect if 0
        if capacity == 0:
            if b.id in _capacity_cache:
                capacity = _capacity_cache[b.id]
            else:
                capacity = autodetect_vram(b.id, b.type)
                if capacity > 0:
                    _capacity_cache[b.id] = capacity
        
        usage = 0
        
        url = f"http://{b.ip}:{b.port}"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{url}/api/ps")
                if resp.status_code == 200:
                    for m in resp.json().get('models', []):
                        usage += m.get('size_vram', m.get('size', 0))
            except:
                pass
        
        # Avoid negative free if capacity is 0/unknown
        if capacity == 0:
            free = 0
        else:
            free = capacity - usage
        
        stats[url] = {
            "type": b.type,
            "free": free, 
            "capacity": capacity,
            "usage": usage,
            "details": {"free_vram_mb": free / (1024*1024)} 
        }
    
    return stats
