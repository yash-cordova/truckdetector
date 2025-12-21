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
    
    async def get_human_violation_analytics(self, year: int, month: int) -> Dict[str, Any]:
        """
        Get Human Violation analytics (DockStatus = RED) for a given month.
        Returns day-wise count of RED status occurrences.
        
        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
        
        Returns:
            Dictionary with day-wise counts
        """
        try:
            if dock1_collection is None:
                logger.warning("MongoDB collection not available")
                return {"error": "Database connection not available"}
            
            # Create date range for the month
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)
            
            # Query documents with RED status in the given month
            query = {
                'dock_id': 'dock1',
                'dock_status': 'RED',
                'timestamp': {
                    '$gte': start_date,
                    '$lt': end_date
                }
            }
            
            # Get all RED status documents
            red_docs = list(dock1_collection.find(query).sort('timestamp', 1))
            
            # Group by day and count (only days with actual data)
            day_wise_count = {}
            for doc in red_docs:
                timestamp = doc.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif not isinstance(timestamp, datetime):
                    continue  # Skip if timestamp is not valid
                
                # Only include if timestamp is within the month range
                if start_date <= timestamp < end_date:
                    day_key = timestamp.strftime('%Y-%m-%d')
                    day_wise_count[day_key] = day_wise_count.get(day_key, 0) + 1
            
            total_violations = len(red_docs)
            
            return {
                'success': True,
                'year': year,
                'month': month,
                'total_violations': total_violations,
                'day_wise_count': day_wise_count,  # Only days with actual data
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting human violation analytics: {str(e)}")
            raise
    
    async def get_idle_condition_analytics(self, year: int, month: int) -> Dict[str, Any]:
        """
        Get Idle Condition analytics for a given month.
        Calculates total used time (in minutes) when vehicle_status was "placed" 
        until it changed to "not_placed", grouped by day.
        
        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
        
        Returns:
            Dictionary with day-wise total used minutes (when truck is in placed state)
        """
        try:
            if dock1_collection is None:
                logger.warning("MongoDB collection not available")
                return {"error": "Database connection not available"}
            
            # Create date range for the month
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)
            
            # Get all documents for the month, sorted by timestamp
            query = {
                'dock_id': 'dock1',
                'timestamp': {
                    '$gte': start_date,
                    '$lt': end_date
                }
            }
            
            docs = list(dock1_collection.find(query).sort('timestamp', 1))
            
            if not docs:
                # No data for this month - return empty
                return {
                    'success': True,
                    'year': year,
                    'month': month,
                    'total_used_minutes': 0.0,
                    'day_wise_used_minutes': {},  # Empty - no data
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate used time periods (when vehicle_status is "placed")
            # Track transitions between consecutive documents - ONLY within the month
            day_wise_used_minutes = {}
            used_start_time = None
            prev_vehicle_status = None
            
            # Get the document just before the month to know the initial state
            prev_doc_query = {
                'dock_id': 'dock1',
                'timestamp': {'$lt': start_date}
            }
            prev_doc = dock1_collection.find_one(prev_doc_query, sort=[('timestamp', -1)])
            if prev_doc:
                prev_vehicle_status = prev_doc.get('vehicle_status', 'not_placed')
                # If previous status was "placed", we need to start tracking from month start
                if prev_vehicle_status == 'placed':
                    used_start_time = start_date
            
            # Filter docs to only those within the month
            month_docs = []
            for doc in docs:
                timestamp = doc.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif not isinstance(timestamp, datetime):
                    continue
                
                if start_date <= timestamp < end_date:
                    month_docs.append(doc)
            
            for i, doc in enumerate(month_docs):
                timestamp = doc.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif not isinstance(timestamp, datetime):
                    continue
                
                vehicle_status = doc.get('vehicle_status', 'not_placed')
                
                # Detect transition from "not_placed" to "placed" (start of used period)
                if prev_vehicle_status == 'not_placed' and vehicle_status == 'placed':
                    # Transition: not_placed -> placed (start tracking used time)
                    used_start_time = timestamp
                
                # Detect transition from "placed" to "not_placed" (end of used period)
                elif prev_vehicle_status == 'placed' and vehicle_status == 'not_placed' and used_start_time is not None:
                    # Transition: placed -> not_placed (end tracking used time)
                    # Only calculate time within the month range
                    end_time = min(timestamp, end_date - timedelta(seconds=1))
                    start_time = max(used_start_time, start_date)
                    
                    if start_time < end_time:
                        # Distribute used time across days (only within month)
                        current_time = start_time
                        while current_time < end_time:
                            day_key = current_time.strftime('%Y-%m-%d')
                            day_end = datetime.combine(current_time.date(), datetime.max.time().replace(microsecond=0))
                            
                            # Don't go beyond end_time or end of month
                            actual_end = min(day_end, end_time)
                            duration = (actual_end - current_time).total_seconds() / 60
                            
                            if duration > 0:
                                if day_key not in day_wise_used_minutes:
                                    day_wise_used_minutes[day_key] = 0
                                day_wise_used_minutes[day_key] += duration
                            
                            # Move to next day
                            current_time = day_end + timedelta(seconds=1)
                            if current_time >= end_time:
                                break
                    
                    # Reset used tracking
                    used_start_time = None
                
                # Update previous state for next iteration
                prev_vehicle_status = vehicle_status
            
            # Handle used period that extends to last document (still placed at last document)
            # BUT only calculate until the last document's timestamp, NOT beyond
            if used_start_time is not None and month_docs:
                last_doc = month_docs[-1]
                last_timestamp = last_doc.get('timestamp')
                if isinstance(last_timestamp, str):
                    last_timestamp = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                
                # CRITICAL: Only calculate until the last document's timestamp
                # Do NOT extrapolate to end of day or end of month
                end_time = last_timestamp  # Use exact timestamp, not end of day
                start_time = max(used_start_time, start_date)
                
                # Only calculate if we have a valid time range
                if start_time < end_time and start_time < end_date:
                    # Calculate duration directly (no day-by-day distribution needed)
                    # Since we're only going to the last document's timestamp
                    duration_minutes = (end_time - start_time).total_seconds() / 60
                    
                    if duration_minutes > 0:
                        # Get the day key for the start time
                        day_key = start_time.strftime('%Y-%m-%d')
                        
                        # If it spans multiple days, distribute it
                        if start_time.date() == end_time.date():
                            # Same day - simple calculation
                            if day_key not in day_wise_used_minutes:
                                day_wise_used_minutes[day_key] = 0
                            day_wise_used_minutes[day_key] += duration_minutes
                        else:
                            # Spans multiple days - distribute
                            current_time = start_time
                            while current_time < end_time:
                                day_key = current_time.strftime('%Y-%m-%d')
                                day_end = datetime.combine(current_time.date(), datetime.max.time().replace(microsecond=0))
                                
                                # Don't go beyond end_time
                                actual_end = min(day_end, end_time)
                                duration = (actual_end - current_time).total_seconds() / 60
                                
                                if duration > 0:
                                    if day_key not in day_wise_used_minutes:
                                        day_wise_used_minutes[day_key] = 0
                                    day_wise_used_minutes[day_key] += duration
                                
                                current_time = day_end + timedelta(seconds=1)
                                if current_time >= end_time:
                                    break
            
            # Only return days with actual used time (no zero values)
            # Round values for days that have data
            filtered_day_wise = {}
            for day_key, minutes in day_wise_used_minutes.items():
                if minutes > 0:  # Only include days with actual used time
                    filtered_day_wise[day_key] = round(minutes, 2)
            
            total_used_minutes = sum(day_wise_used_minutes.values())
            
            return {
                'success': True,
                'year': year,
                'month': month,
                'total_used_minutes': round(total_used_minutes, 2),
                'day_wise_used_minutes': filtered_day_wise,  # Only days with actual data
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting idle condition analytics: {str(e)}")
            raise

# Global dock service instance
dock_service = DockService()
