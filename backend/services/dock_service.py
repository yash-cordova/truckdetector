import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pymongo import MongoClient
from backend.db.mongo import dock1_collection, client
from backend.models.dock import DockData, DockUpdate, VehicleStatus, HumanPresence, DockStatus

logger = logging.getLogger(__name__)

class DockService:
    def __init__(self):
        self.change_stream = None
        self.dock_data = {}
        self.websocket_connections = []
        
    async def start_change_stream(self):
        """Start polling mechanism for dock1 collection (since Change Streams require replica set)"""
        try:
            # Check if MongoDB connection exists
            if dock1_collection is None:
                logger.error("MongoDB connection not available. Polling mechanism will not start.")
                return
            
            logger.info("Started polling mechanism for dock1 (Change Streams not available in standalone MongoDB)")
            
            last_count = 0
            while True:
                try:
                    # Poll for changes by counting documents
                    if dock1_collection is not None:
                        current_count = dock1_collection.count_documents({'dock_id': 'dock1'})
                        
                        if current_count != last_count:
                            logger.info(f"Dock1 data changed - count: {last_count} -> {current_count}")
                            await self.handle_polling_change()
                            last_count = current_count
                    else:
                        logger.warning("MongoDB collection not available, skipping poll")
                    
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
        
        # Keep dock_status from doc_data if provided, otherwise None (NA)
        dock_status = doc_data.get('dock_status')
        
        return {
            'vehicle_status': vehicle_status,
            'human_presence': human_presence,
            'dock_status': dock_status,  # Keep existing or None (NA)
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
            if dock1_collection is None:
                logger.warning("MongoDB collection not available, returning empty status")
                return {}
            
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
                    'dock_status': str(latest_doc.get('dock_status')) if latest_doc.get('dock_status') is not None else None,
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
            if dock1_collection is None:
                logger.warning("MongoDB collection not available, returning empty list")
                return []
            
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
                    'dock_status': str(doc.get('dock_status')) if doc.get('dock_status') is not None else None,
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
    
    async def update_dock_status(self, dock_id: str, update_data: DockUpdate) -> Dict[str, Any]:
        """
        Update dock status in the database.
        Creates a new document with timestamp for history tracking.
        
        Args:
            dock_id: Dock identifier (e.g., 'dock1')
            update_data: DockUpdate model with fields to update
        
        Returns:
            Dictionary with updated dock status
        """
        try:
            if dock1_collection is None:
                logger.error("MongoDB collection not available")
                raise Exception("Database connection not available")
            
            # Get current status to merge with updates
            current_status = await self.get_current_dock_status()
            
            # Prepare update data
            current_time = datetime.now()
            new_doc = {
                'dock_id': dock_id,
                'timestamp': current_time,
                'last_updated': current_time,
            }
            
            # Merge current status with updates
            if current_status:
                # Use current values as defaults
                new_doc['vehicle_status'] = current_status.get('vehicle_status', 'not_placed')
                new_doc['human_presence'] = current_status.get('human_presence', 'not_present')
                new_doc['dock_status'] = current_status.get('dock_status')  # Keep existing or None (NA)
                new_doc['ideal_time_minutes'] = current_status.get('ideal_time_minutes', 0)
                new_doc['non_ideal_time_minutes'] = current_status.get('non_ideal_time_minutes', 0)
                new_doc['notes'] = current_status.get('notes', '')
            else:
                # Default values if no current status
                new_doc['vehicle_status'] = 'not_placed'
                new_doc['human_presence'] = 'not_present'
                new_doc['dock_status'] = None  # NA if not provided
                new_doc['ideal_time_minutes'] = 0
                new_doc['non_ideal_time_minutes'] = 0
                new_doc['notes'] = ''
            
            # Apply updates from request
            if update_data.vehicle_status is not None:
                new_doc['vehicle_status'] = update_data.vehicle_status.value
            if update_data.human_presence is not None:
                new_doc['human_presence'] = update_data.human_presence.value
            if update_data.dock_status is not None:
                new_doc['dock_status'] = update_data.dock_status.value
            # If dock_status not provided, keep existing value or None (NA)
            elif 'dock_status' not in new_doc:
                new_doc['dock_status'] = None
            if update_data.notes is not None:
                new_doc['notes'] = update_data.notes
            
            # Calculate time metrics if status changed
            if current_status:
                # Calculate time difference
                last_updated_str = current_status.get('last_updated')
                if isinstance(last_updated_str, str):
                    try:
                        last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
                    except:
                        last_updated = current_time
                else:
                    last_updated = current_time
                
                time_diff = (current_time - last_updated).total_seconds() / 60  # minutes
                
                # Determine if previous state was ideal
                prev_vehicle = current_status.get('vehicle_status', 'not_placed')
                prev_human = current_status.get('human_presence', 'not_present')
                was_ideal = prev_vehicle == 'not_placed' and prev_human == 'not_present'
                
                # Update time metrics
                prev_ideal_time = current_status.get('ideal_time_minutes', 0)
                prev_non_ideal_time = current_status.get('non_ideal_time_minutes', 0)
                
                if was_ideal:
                    new_doc['ideal_time_minutes'] = prev_ideal_time + time_diff
                    new_doc['non_ideal_time_minutes'] = prev_non_ideal_time
                else:
                    new_doc['ideal_time_minutes'] = prev_ideal_time
                    new_doc['non_ideal_time_minutes'] = prev_non_ideal_time + time_diff
            else:
                # First entry - no time metrics yet
                new_doc['ideal_time_minutes'] = 0
                new_doc['non_ideal_time_minutes'] = 0
            
            # Insert new document (for history tracking)
            result = dock1_collection.insert_one(new_doc)
            
            logger.info(f"Updated dock status for {dock_id}: vehicle={new_doc.get('vehicle_status')}, human={new_doc.get('human_presence')}, status={new_doc.get('dock_status')}")
            
            # Trigger WebSocket broadcast
            await self.handle_polling_change()
            
            # Return formatted response
            return {
                '_id': str(result.inserted_id),
                'dock_id': new_doc['dock_id'],
                'vehicle_status': new_doc['vehicle_status'],
                'human_presence': new_doc['human_presence'],
                'dock_status': new_doc.get('dock_status'),  # Can be None (NA)
                'ideal_time_minutes': new_doc['ideal_time_minutes'],
                'non_ideal_time_minutes': new_doc['non_ideal_time_minutes'],
                'last_updated': new_doc['last_updated'].isoformat(),
                'timestamp': new_doc['timestamp'].isoformat(),
                'notes': new_doc['notes']
            }
            
        except Exception as e:
            logger.error(f"Error updating dock status: {str(e)}")
            raise

# Global dock service instance
dock_service = DockService()
