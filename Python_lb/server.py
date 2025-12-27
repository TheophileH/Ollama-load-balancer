from app.main import app
from app.discovery import registry
import uvicorn
import os

@app.on_event("startup")
async def startup_event():
    print("Starting discovery service...")
    registry.start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 11434))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
