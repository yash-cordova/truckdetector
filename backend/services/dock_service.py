import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pymongo import MongoClient
from backend.db.mongo import dock1_collection, client
from backend.models.dock import DockData, VehicleStatus, HumanPresence, DockStatus

logger = logging.getLogger(__name__)

class DockService:
    def __init__(self):
        self.change_stream = None
        self.dock_data = {}
        self.websocket_connections = []
        
    async def start_change_stream(self):
        """Start polling mechanism for dock1 collection (since Change Streams require replica set)"""
        try:
            logger.info("Started polling mechanism for dock1 (Change Streams not available in standalone MongoDB)")
            
            last_count = 0
            while True:
                try:
                    # Poll for changes by counting documents
                    current_count = dock1_collection.count_documents({'dock_id': 'dock1'})
                    
                    if current_count != last_count:
                        logger.info(f"Dock1 data changed - count: {last_count} -> {current_count}")
                        await self.handle_polling_change()
                        last_count = current_count
                    
                    # Poll every 2 seconds
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error in polling loop: {str(e)}")
                    await asyncio.sleep(5)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Error in polling mechanism: {str(e)}")
    
    async def handle_polling_change(self):
        """Handle polling-based change detection"""
        try:
            # Get current status and recent data
            current_status = await self.get_current_dock_status()
            recent_data = await self.get_recent_dock_data(limit=10)
            
            # Prepare update message with both current status and recent data
            update_message = {
                "type": "dock_update",
                "operation": "polling_update",
                "dock_id": "dock1",
                "current_status": current_status,
                "recent_data": recent_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast to all WebSocket connections
            await self.broadcast_to_websockets(update_message)
            
            logger.info(f"Polling update sent - {len(recent_data)} recent entries")
            
        except Exception as e:
            logger.error(f"Error handling polling change: {str(e)}")
            
    async def handle_change_event(self, change: Dict[str, Any]):
        """Handle MongoDB change events and broadcast to WebSocket clients"""
        try:
            operation_type = change.get('operationType')
            full_document = change.get('fullDocument')
            document_key = change.get('documentKey', {}).get('_id')
            
            # Get current status and recent data
            current_status = await self.get_current_dock_status()
            recent_data = await self.get_recent_dock_data(limit=10)
            
            # Prepare update message with both current status and recent data
            update_message = {
                "type": "dock_update",
                "operation": operation_type,
                "dock_id": "dock1",
                "current_status": current_status,
                "recent_data": recent_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast to all WebSocket connections
            await self.broadcast_to_websockets(update_message)
            
            logger.info(f"Processed {operation_type} event for dock1 - sent {len(recent_data)} recent entries")
            
        except Exception as e:
            logger.error(f"Error handling change event: {str(e)}")
            
    async def calculate_time_metrics(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ideal and non-ideal time based on dock status"""
        if not doc_data:
            return {}
            
        current_time = datetime.now()
        
        # Get current dock status
        vehicle_status = doc_data.get('vehicle_status', 'not_placed')
        human_presence = doc_data.get('human_presence', 'not_present')
        
        # Determine if dock is ideal (no vehicle, no human)
        is_ideal = vehicle_status == 'not_placed' and human_presence == 'not_present'
        
        # Calculate time since last update
        last_updated = doc_data.get('last_updated')
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
        elif not isinstance(last_updated, datetime):
            last_updated = current_time
            
        time_diff = (current_time - last_updated).total_seconds() / 60  # minutes
        
        # Update time metrics
        ideal_time = doc_data.get('ideal_time_minutes', 0)
        non_ideal_time = doc_data.get('non_ideal_time_minutes', 0)
        
        if is_ideal:
            ideal_time += time_diff
        else:
            non_ideal_time += time_diff
            
        # Update document with new time metrics
        dock1_collection.update_one(
            {'_id': doc_data['_id']},
            {
                '$set': {
                    'ideal_time_minutes': ideal_time,
                    'non_ideal_time_minutes': non_ideal_time,
                    'last_updated': current_time
                }
            }
        )
        
        return {
            'vehicle_status': vehicle_status,
            'human_presence': human_presence,
            'dock_status': 'idle' if is_ideal else 'occupied',
            'ideal_time_minutes': ideal_time,
            'non_ideal_time_minutes': non_ideal_time,
            'last_updated': current_time.isoformat(),
            'notes': doc_data.get('notes', '')
        }
        
    async def broadcast_to_websockets(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections"""
        for websocket in self.websocket_connections[:]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {str(e)}")
                self.websocket_connections.remove(websocket)
    
    async def trigger_update(self):
        """Manually trigger a WebSocket update with current data"""
        try:
            logger.info("Manually triggering WebSocket update")
            await self.handle_polling_change()
        except Exception as e:
            logger.error(f"Error triggering manual update: {str(e)}")
                
    def add_websocket_connection(self, websocket):
        """Add a new WebSocket connection"""
        self.websocket_connections.append(websocket)
        logger.info(f"Added WebSocket connection. Total: {len(self.websocket_connections)}")
        
    def remove_websocket_connection(self, websocket):
        """Remove a WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
        logger.info(f"Removed WebSocket connection. Total: {len(self.websocket_connections)}")
        
    async def get_current_dock_status(self) -> Dict[str, Any]:
        """Get current dock status from database - latest entry sorted by time with all fields"""
        try:
            # Get the latest entry sorted by timestamp
            latest_doc = dock1_collection.find_one(
                {'dock_id': 'dock1'},
                sort=[('timestamp', -1)]
            )
            if latest_doc:
                # Return the complete document with all fields formatted
                return {
                    '_id': str(latest_doc['_id']),
                    'dock_id': str(latest_doc.get('dock_id', 'dock1')),
                    'vehicle_status': str(latest_doc.get('vehicle_status', 'not_placed')),
                    'human_presence': str(latest_doc.get('human_presence', 'not_present')),
                    'dock_status': str(latest_doc.get('dock_status', 'idle')),
                    'ideal_time_minutes': int(latest_doc.get('ideal_time_minutes', 0)),
                    'non_ideal_time_minutes': int(latest_doc.get('non_ideal_time_minutes', 0)),
                    'last_updated': latest_doc.get('last_updated', datetime.now()).isoformat() if isinstance(latest_doc.get('last_updated'), datetime) else str(latest_doc.get('last_updated', datetime.now().isoformat())),
                    'notes': str(latest_doc.get('notes', '')),
                    'timestamp': latest_doc.get('timestamp', datetime.now()).isoformat() if isinstance(latest_doc.get('timestamp'), datetime) else str(latest_doc.get('timestamp', datetime.now().isoformat()))
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting dock status: {str(e)}")
            return {}
    
    async def get_recent_dock_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent dock data entries sorted by time (newest first) with all fields"""
        try:
            # Get recent entries sorted by timestamp (newest first)
            recent_docs = list(dock1_collection.find(
                {'dock_id': 'dock1'},
                sort=[('timestamp', -1)],
                limit=limit
            ))
            
            # Convert ObjectId to string and format all datetime fields
            formatted_data = []
            for doc in recent_docs:
                # Convert ObjectId to string
                doc['_id'] = str(doc['_id'])
                
                # Convert all datetime fields to ISO format
                for field in ['timestamp', 'last_updated']:
                    if field in doc and isinstance(doc[field], datetime):
                        doc[field] = doc[field].isoformat()
                
                # Ensure all fields are included as strings/numbers (not ObjectId)
                formatted_doc = {
                    '_id': str(doc['_id']),
                    'dock_id': str(doc.get('dock_id', 'dock1')),
                    'vehicle_status': str(doc.get('vehicle_status', 'not_placed')),
                    'human_presence': str(doc.get('human_presence', 'not_present')),
                    'dock_status': str(doc.get('dock_status', 'idle')),
                    'ideal_time_minutes': int(doc.get('ideal_time_minutes', 0)),
                    'non_ideal_time_minutes': int(doc.get('non_ideal_time_minutes', 0)),
                    'last_updated': doc.get('last_updated', datetime.now().isoformat()),
                    'notes': str(doc.get('notes', '')),
                    'timestamp': doc.get('timestamp', datetime.now().isoformat())
                }
                formatted_data.append(formatted_doc)
            
            return formatted_data
        except Exception as e:
            logger.error(f"Error getting recent dock data: {str(e)}")
            return []

# Global dock service instance
dock_service = DockService()
