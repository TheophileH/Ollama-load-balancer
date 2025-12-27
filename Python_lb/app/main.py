import httpx
import logging
import random
from fastapi import FastAPI, Request, HTTPException
from app.discovery import registry
from app.proxy import proxy_request
from app.logic import route_request

import os

# Configure Logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
logger = logging.getLogger("main")

app = FastAPI()

@app.get("/api/tags")
async def list_models():
    """Aggregate models from all discovered instances."""
    models_map = {}
    backends = registry.get_instances()
    
    if not backends:
        raise HTTPException(status_code=503, detail="No backends available")

    async with httpx.AsyncClient() as client:
        # improvement: use asyncio.gather for parallel requests
        for b in backends:
            url = f"http://{b.ip}:{b.port}"
            try:
                resp = await client.get(f"{url}/api/tags")
                if resp.status_code == 200:
                    for m in resp.json().get('models', []):
                        # Simple de-duplication by name
                        if m['name'] not in models_map:
                            models_map[m['name']] = m
            except Exception as e:
                logger.error(f"Failed to fetch tags from {b}: {e}")
                
    return {"models": list(models_map.values())}

@app.get("/api/ps")
async def list_running():
    """Aggregate running models from all instances."""
    models = []
    backends = registry.get_instances()
    
    async with httpx.AsyncClient() as client:
        for b in backends:
            url = f"http://{b.ip}:{b.port}"
            try:
                resp = await client.get(f"{url}/api/ps")
                if resp.status_code == 200:
                    # Optional: Tag which backend it's running on?
                    # For standard API compat, keeping it clean, but for debugging handy.
                    items = resp.json().get('models', [])
                    models.extend(items)
            except Exception as e:
                logger.error(f"Failed to fetch ps from {b}: {e}")
            
    return {"models": models}

@app.post("/api/chat")
async def chat(request: Request):
    return await route_request(request, "/api/chat")

@app.post("/api/generate")
async def generate(request: Request):
    return await route_request(request, "/api/generate")

@app.post("/api/embed")
async def embed(request: Request):
    return await route_request(request, "/api/embed")

@app.post("/api/show")
async def show(request: Request):
    return await route_request(request, "/api/show")

@app.post("/api/pull")
async def pull(request: Request):
    # Pull requests should probably target specific backends or all?
    # For now, route request logic will pick one based on strategy.
    # TODO: Sync pulling across replicas?
    return await route_request(request, "/api/pull")

@app.delete("/api/delete")
async def delete(request: Request):
    # Delete needs to happen on shared storage usually.
    # We'll pick ANY backend, assuming shared volume.
    backends = registry.get_instances()
    if not backends:
        raise HTTPException(status_code=503, detail="No backends")
    
    target = f"http://{backends[0].ip}:{backends[0].port}"
    return await proxy_request(request, target, "/api/delete")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
async def catch_all(path: str, request: Request):
    # Default proxy for version, root, etc.
    backends = registry.get_instances()
    if not backends:
         raise HTTPException(status_code=503, detail="No backends available")
    
    # Random load balance
    selected = random.choice(backends)
    target_url = f"http://{selected.ip}:{selected.port}"
    
    # Path handling: if path is empty (root), don't add slash? 
    # FastAPI path acts weird on root.
    # If request is just "/", path is "". Url should be target_url/
    
    if path and not path.startswith("/"):
        path = "/" + path
    elif not path:
        path = "/"

    return await proxy_request(request, target_url, path)

