"""Script to run the FastAPI application using uvicorn."""

import uvicorn
import sys
import os
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import settings from config
from backend.core.config import settings

# Suppress uvicorn access logs (WebSocket connection attempts, HTTP requests, etc.)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)  # Only show errors, not INFO
logging.getLogger("uvicorn.access").disabled = True  # Completely disable access logs

if __name__ == "__main__":
    # Configuration - use config file settings, but allow environment variable override
    host = os.getenv("HOST", settings.SERVER_HOST)
    port = int(os.getenv("PORT", settings.SERVER_PORT))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    # Workers: Number of parallel processes to handle requests
    # - 1 worker = Single process (good for development, low traffic)
    # - 2-4 workers = Multiple processes (good for production, handles more concurrent requests)
    # - More workers = Better performance but uses more memory (each worker loads the entire app)
    # Note: Workers don't work with reload mode (reload only works with 1 worker)
    workers = int(os.getenv("WORKERS", 1))
    
    print(f"Starting server on {host}:{port}")
    print(f"Reload: {reload}")
    print(f"Workers: {workers} {'(reload mode, using 1 worker)' if reload and workers > 1 else ''}")
    print("-" * 50)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Reload doesn't work with multiple workers
        log_level="info",
        access_log=False  # Disable access logs (suppresses WebSocket connection attempts, 403 errors, etc.)
    )

