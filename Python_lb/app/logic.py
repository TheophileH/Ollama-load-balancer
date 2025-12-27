import os
import random
from fastapi import Request, HTTPException
from app.config import OLLAMA_AMD_URL, OLLAMA_NVIDIA_URL 
from app.discovery import registry
from app.utils import get_model_size, get_vram_stats, get_running_models
from app.proxy import forward_request

async def route_request(request: Request, path: str):
    body = await request.json()
    if body.get("model"):
        model_field = "model"
    elif body.get("name"):
        model_field = "name"
    else:
        raise HTTPException(status_code=400, detail="Model name required")

    model_name = body[model_field]
    target_type = None

    # 0. Prefix Directive (e.g. "nvidia/model" or "cuda/model")
    PREFIX_MAP = {
        "nvidia/": "nvidia",
        "cuda/": "nvidia",
        "amd/": "amd",
        "rocm/": "amd"
    }

    for prefix, directive in PREFIX_MAP.items():
        if model_name.startswith(prefix):
            print(f"Prefix directive: {prefix} -> {directive}")
            target_type = directive
            # Strip prefix
            body[model_field] = model_name[len(prefix):]
            break

    # 1. Suffix Directive (e.g. "model@amd")
    if not target_type and "@" in model_name:
        base_name, directive = model_name.rsplit("@", 1)
        if directive in ["amd", "nvidia"]:
            print(f"Manual directive: {directive}")
            target_type = directive
            body[model_field] = base_name

    # ... (Selection Logic) ...
    
    # 2. Determine Strategy & Candidates
    # If explicit target_type is set (by prefix/suffix), we respect it.
    # Otherwise we check strategy.
    
    candidates = []
    if target_type:
        candidates = registry.get_instances(target_type)
    else:
        candidates = registry.get_instances() # All

    if not candidates:
        raise HTTPException(status_code=503, detail="No backend Ollama instances available")

    selected = None
    
    # Resolve Strategy for Fallback
    strategy = request.headers.get("X-Load-Balancing-Strategy")
    if not strategy:
        strategy = os.getenv("LOAD_BALANCING_STRATEGY", "round_robin")
    strategy = strategy.lower()

    # Check for Affinity / Smart Load Balancing
    enable_affinity = os.getenv("OLLAMA_ENABLE_AFFINITY", "false").lower() == "true"
    
    if enable_affinity:
        # Strategy:
        # 1. Prefer node with model loaded
        # 2. Prefer node with least active requests
        
        print(f"Applying Affinity Strategy for {model_name}...")
        
        # Get loaded models map {url: [models]}
        # Note: This might add latency. In high-perf, cache this or track locally.
        loaded_map = await get_running_models()
        
        scored_candidates = []
        for c in candidates:
            url = f"http://{c.ip}:{c.port}"
            loaded_models = loaded_map.get(url, [])
            
            # Simple match (exact or name without tag)
            is_loaded = any(m == model_name or m.split(":")[0] == model_name.split(":")[0] for m in loaded_models)
            
            # Score: (Has Model?, -Active Requests)
            # We want True > False, and Low Active > High Active
            score = (1 if is_loaded else 0, -c.active_requests)
            scored_candidates.append((score, c))
            
            print(f"  - Node {c.id} ({c.type}): Loaded={is_loaded}, Active={c.active_requests} -> Score={score}")

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_candidate = scored_candidates[0]
        
        # HYBRID STRATEGY FIX:
        # Only select here if the model is ACTUALLY loaded.
        # If not loaded (score[0] == 0), we fall back to the configured strategy (VRAM/RoundRobin)
        # to ensure smart placement of new models.
        if best_score[0] == 1:
             selected = best_candidate
             print(f"Affinity Hit! Routing to {selected.id}")
        else:
             print("Affinity Miss. Falling back to configured strategy...")
        
    if not selected and strategy == "vram":
        # VRAM Logic (Existing)
        # Select best free VRAM
        stats_map = await get_vram_stats()
        max_free = -1
        
        for c in candidates:
             s = stats_map.get(f"http://{c.ip}:{c.port}")
             if s:
                 free = s.get("details", {}).get("free_vram_mb", 0)
                 if free > max_free:
                     max_free = free
                     selected = c
        
        if not selected and candidates:
            selected = random.choice(candidates)
            
    if not selected:
        # Round Robin / Random fallback
        selected = random.choice(candidates)

    # 3. Forward Request & Track Load
    target_url = f"http://{selected.ip}:{selected.port}"
    print(f"Selected backend: {selected} (Active Reqs: {selected.active_requests})")
    
    # Increment Load
    selected.active_requests += 1
    
    def on_complete():
        selected.active_requests -= 1
        print(f"Request finished for {selected.id}. Active Reqs: {selected.active_requests}")

    return await forward_request(request, target_url, path, body, on_complete=on_complete)

