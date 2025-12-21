import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, WebSocketException
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.auth import auth_router
from backend.services.dock_service import dock_service
from backend.models.dock import DockUpdate
from backend.core.config import settings
from datetime import datetime
from typing import List
import asyncio
import uvicorn

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Suppress uvicorn access logs (WebSocket connection attempts, HTTP requests, etc.)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").disabled = True

logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the auth routes
app.include_router(auth_router)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_health_check: datetime = datetime.now()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast_health(self, message: dict):
        self.last_health_check = datetime.now()
        for connection in self.active_connections[:]:  # Create a copy to avoid modification during iteration
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                self.disconnect(connection)
            except Exception as e:
                logger.error(f"Error broadcasting health: {str(e)}")
                self.disconnect(connection)

manager = ConnectionManager()

# Start change stream task
@app.on_event("startup")
async def startup_event():
    """Start the MongoDB change stream when the application starts"""
    asyncio.create_task(dock_service.start_change_stream())
    logger.info("Started dock service change stream")

# Dock WebSocket endpoint
@app.websocket("/ws/dock1")
async def websocket_dock1(websocket: WebSocket):
    await websocket.accept()
    dock_service.add_websocket_connection(websocket)
    
    try:
        # Send current dock status and recent data immediately
        current_status = await dock_service.get_current_dock_status()
        recent_data = await dock_service.get_recent_dock_data(limit=10)
        
        await websocket.send_json({
            "type": "initial_status",
            "dock_id": "dock1",
            "current_status": current_status,
            "recent_data": recent_data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and listen for any messages
        while True:
            try:
                # Wait for any message from client (ping/pong)
                data = await websocket.receive_text()
                # Echo back or handle client messages if needed
                await websocket.send_text(f"Received: {data}")
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in dock1 WebSocket: {str(e)}")
                break

    except WebSocketDisconnect:
        dock_service.remove_websocket_connection(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
        dock_service.remove_websocket_connection(websocket)

# WebSocket endpoints for dock2-5 (silently reject - these docks are not implemented)
@app.websocket("/ws/dock2")
@app.websocket("/ws/dock3")
@app.websocket("/ws/dock4")
@app.websocket("/ws/dock5")
async def websocket_dock_not_implemented(websocket: WebSocket):
    """Silently reject WebSocket connections for unimplemented docks"""
    # Reject connection before accepting to avoid triggering access logs
    raise WebSocketException(code=1003, reason="Dock not implemented")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "healthy",
            "database": "connected"
        },
        "system": {
            "version": "2.1.0",
            "uptime": "running",
            "connections": len(manager.active_connections)
        },
        "message": "Authentication service is running"
    }

# WebSocket health endpoint
@app.websocket("/ws/health")
async def websocket_health(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send health data every 5 seconds
            health_data = {
                "status": "online",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "api": "healthy",
                    "database": "connected"
                },
                "system": {
                    "version": "2.1.0",
                    "uptime": "running",
                    "connections": len(manager.active_connections)
                },
                "message": "Authentication service is running"
            }
            await manager.broadcast_health(health_data)
            
            # Sleep for 5 seconds before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
        manager.disconnect(websocket)

# REST endpoint to manually trigger WebSocket update
@app.post("/trigger-dock-update")
async def trigger_dock_update():
    """Manually trigger a WebSocket update for dock1 data"""
    try:
        await dock_service.trigger_update()
        return {
            "success": True,
            "message": "WebSocket update triggered successfully",
            "timestamp": datetime.now().isoformat(),
            "active_connections": len(dock_service.websocket_connections)
        }
    except Exception as e:
        logger.error(f"Error triggering dock update: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger update: {str(e)}")

# REST endpoint to get current dock status
@app.get("/dock1/status")
async def get_dock_status():
    """Get current dock1 status"""
    try:
        current_status = await dock_service.get_current_dock_status()
        recent_data = await dock_service.get_recent_dock_data(limit=5)
        
        return {
            "success": True,
            "dock_id": "dock1",
            "current_status": current_status,
            "recent_data": recent_data,
            "timestamp": datetime.now().isoformat(),
            "total_entries": len(recent_data)
        }
    except Exception as e:
        logger.error(f"Error getting dock status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dock status: {str(e)}")

# REST endpoint to update dock status
@app.post("/dock1/status")
async def update_dock_status(update_data: DockUpdate):
    """
    Update dock1 status.
    Accepts partial updates - only provided fields will be updated.
    Creates a new document with timestamp for history tracking.
    """
    try:
        updated_status = await dock_service.update_dock_status("dock1", update_data)
        
        return {
            "success": True,
            "message": "Dock status updated successfully",
            "dock_id": "dock1",
            "updated_status": updated_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating dock status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update dock status: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT)
