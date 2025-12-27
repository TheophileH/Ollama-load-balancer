import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_AMD_URL = os.getenv("OLLAMA_AMD_URL")
OLLAMA_NVIDIA_URL = os.getenv("OLLAMA_NVIDIA_URL")

from app.discovery import registry

# Initialize Registry (will start thread on import or handled in main)
# We don't export static BACKENDS anymore. Logic should verify registry.

