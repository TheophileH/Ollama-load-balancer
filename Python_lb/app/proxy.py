import httpx
from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse

async def forward_request(request: Request, target_url: str, path: str, body: dict, on_complete=None):
    client = httpx.AsyncClient()
    url = f"{target_url}{path}"
    
    # Strip Headers
    forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in ["content-length", "host"]}
    
    try:
        req = client.build_request(
            request.method,
            url,
            json=body,
            headers=forward_headers,
            timeout=None
        )
        
        r = await client.send(req, stream=True)
        
        async def cleanup():
            await client.aclose()
            if on_complete:
                if isinstance(on_complete, list):
                    for cb in on_complete: cb()
                else:
                    on_complete()

        return StreamingResponse(
            r.aiter_raw(),
            status_code=r.status_code,
            headers=r.headers,
            background=cleanup
        )
    except Exception as e:
        print(f"Forward request failed: {e}")
        await client.aclose()
        raise HTTPException(status_code=500, detail=str(e))

async def proxy_request(request: Request, base_url: str, path: str):
    client = httpx.AsyncClient()
    url = f"{base_url}{path}"
    
    content = await request.body()
    
    try:
        req = client.build_request(
            request.method,
            url,
            content=content,
            headers=request.headers.raw,
            timeout=None
        )
        
        r = await client.send(req, stream=True)
        return StreamingResponse(
            r.aiter_raw(),
            status_code=r.status_code,
            headers=r.headers,
            background=client.aclose
        )
    except Exception as e:
        print(f"Proxy request failed: {e}")
        await client.aclose()
        # If headers/status not available, 500
        raise HTTPException(status_code=500, detail=str(e))
