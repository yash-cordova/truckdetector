"""Truck monitoring logic: state management, violation detection, and status updates"""
import time


class TruckMonitor:
    """Handles truck state tracking, violation detection, and status management"""
    
    def __init__(self, parking_duration_seconds=5.0):
        self.truck_state = "NO_TRUCK"  # NO_TRUCK, TRUCK_DETECTED, TOUCHED_LINE, PARKED
        self.last_truck_position = None
        self.parking_line_y = None
        self.parking_timer_start = None
        self.parking_duration_seconds = parking_duration_seconds
        
    def update_truck_state(self, trucks, roi, parking_line_y):
        """Update truck state based on detection and position"""
        self.parking_line_y = parking_line_y
        
        if not trucks:
            # No truck detected
            self.truck_state = "NO_TRUCK"
            self.last_truck_position = None
            self.parking_timer_start = None
            return
        
        # Get the largest truck (most confident)
        truck = max(trucks, key=lambda x: x[4])
        x1, y1, x2, y2 = truck[:4]
        truck_center_y = (y1 + y2) / 2
        
        # Check if truck is actually in ROI before updating state
        # Only update truck state if truck bounding box touches or overlaps ROI
        if roi:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi
            # Check if truck bbox overlaps with ROI
            truck_in_roi = not (x2 < roi_x1 or x1 > roi_x2 or y2 < roi_y1 or y1 > roi_y2)
            
            if not truck_in_roi:
                self.truck_state = "NO_TRUCK"
                self.last_truck_position = None
                self.parking_timer_start = None
                return
        
        if self.parking_line_y is None:
            return
        
        # Check if truck bounding box has touched the parking line
        # Truck touches when bottom edge crosses the line
        has_touched = y2 >= self.parking_line_y
        
        current_time = time.time()
        
        if not has_touched:
            # Truck is detected but hasn't touched line yet
            self.truck_state = "TRUCK_DETECTED"
            self.parking_timer_start = None
        else:
            # Truck has touched the line
            if self.truck_state in ["NO_TRUCK", "TRUCK_DETECTED"]:
                # Just touched the line - start timer
                self.truck_state = "TOUCHED_LINE"
                self.parking_timer_start = current_time
            elif self.truck_state == "TOUCHED_LINE":
                # Check if parking duration has passed
                if self.parking_timer_start is not None:
                    elapsed = current_time - self.parking_timer_start
                    if elapsed >= self.parking_duration_seconds:
                        self.truck_state = "PARKED"
                else:
                    # Timer was lost, restart it
                    self.parking_timer_start = current_time
        
        self.last_truck_position = truck_center_y
    
    def check_person_violations(self, persons, trucks, roi, roi_corners=None):
        """Check for person violations - returns True if violation exists
        
        IMPORTANT: Only checks bottom-center dot (feet position) of person bounding box.
        Even if upper body is in ROI, if bottom-center dot is NOT in ROI, it's NOT a violation.
        This handles side-angle camera views where walkway is behind ROI boundary.
        """
        # NO VIOLATION if no truck detected or truck state is NO_TRUCK
        if not trucks or len(trucks) == 0 or self.truck_state == "NO_TRUCK":
            return False
        
        # NO VIOLATION if truck is PARKED
        if self.truck_state == "PARKED":
            return False
        
        # Check if ROI is set
        if not roi:
            return False
        
        # Get ROI as polygon (5 corners) for accurate checking
        use_polygon = roi_corners is not None and len(roi_corners) >= 3
        
        # NO VIOLATION if no persons detected
        if not persons or len(persons) == 0:
            return False
        
        # Check each person: ONLY bottom-center dot being in ROI counts as violation
        for person in persons:
            px1, py1, px2, py2 = person[:4]
            # Use bottom-center point of person bounding box (feet position)
            person_bottom_center_x = (px1 + px2) / 2
            person_bottom_y = py2
            
            # Check if person's bottom-center point is in ROI polygon
            if use_polygon:
                person_bottom_in_roi = self._point_in_polygon(person_bottom_center_x, person_bottom_y, roi_corners)
            else:
                # Fallback to bounding box check
                roi_x1, roi_y1, roi_x2, roi_y2 = roi
                person_bottom_in_roi = (roi_x1 <= person_bottom_center_x <= roi_x2 and 
                            roi_y1 <= person_bottom_y <= roi_y2)
            
            if person_bottom_in_roi:
                return True
        
        return False
    
    def _point_in_polygon(self, x, y, polygon):
        """Check if point (x, y) is inside polygon using ray casting algorithm"""
        if len(polygon) < 3:
            return False
        
        try:
            inside = False
            j = len(polygon) - 1
            for i in range(len(polygon)):
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                
                # Check if ray intersects with edge (avoid division by zero)
                if ((yi > y) != (yj > y)) and (yj != yi):  # yj != yi to avoid division by zero
                    if (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside = not inside
                j = i
            
            return inside
        except Exception as e:
            print(f"Error in point_in_polygon: {e}")
            return False
    
    def check_person_or_forklift_in_roi(self, persons, forklifts, roi):
        """Check if any person or forklift is in ROI"""
        if not roi:
            return False
        
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        
        # Check persons (using bottom-center dot)
        for person in persons:
            px1, py1, px2, py2 = person[:4]
            person_bottom_center_x = (px1 + px2) / 2
            person_bottom_y = py2
            if (roi_x1 <= person_bottom_center_x <= roi_x2 and 
                roi_y1 <= person_bottom_y <= roi_y2):
                return True
        
        # Check forklifts (using center point)
        for forklift in forklifts:
            fx1, fy1, fx2, fy2 = forklift[:4]
            forklift_center_x = (fx1 + fx2) / 2
            forklift_center_y = (fy1 + fy2) / 2
            if (roi_x1 <= forklift_center_x <= roi_x2 and 
                roi_y1 <= forklift_center_y <= roi_y2):
                return True
        
        return False
    
    def get_status_info(self, has_violation, has_person_or_forklift_in_roi):
        """Get status text and active light color based on current state"""
        if self.truck_state == "NO_TRUCK":
            if has_person_or_forklift_in_roi:
                active_light = "green"  # Still green if person/forklift but no truck
                status_text = "Status: NO_TRUCK"
            else:
                active_light = "green"
                status_text = "Status: NO_TRUCK - Safe"
        elif self.truck_state == "PARKED":
            active_light = "green"
            status_text = "Status: PARKED - Safe to start unloading"
        elif has_violation:
            # Person's bottom-center dot is in ROI = RED (violation)
            active_light = "red"
            status_text = "Status: TRUCK DETECTED - VIOLATION (Person feet in ROI)"
        elif self.truck_state == "TOUCHED_LINE":
            active_light = "yellow"  # Yellow because no violation (bottom-center dot not in ROI)
            # Show countdown if available
            if self.parking_timer_start is not None:
                elapsed = time.time() - self.parking_timer_start
                remaining = max(0, self.parking_duration_seconds - elapsed)
                status_text = f"Status: TRUCK TOUCHED LINE - Wait {remaining:.1f}s for PARKED"
            else:
                status_text = "Status: TRUCK TOUCHED LINE"
        else:  # TRUCK_DETECTED
            active_light = "yellow"  # Yellow because no violation (bottom-center dot not in ROI)
            status_text = "Status: TRUCK DETECTED - Approaching"
        
        return active_light, status_text
    
    def reset(self):
        """Reset all state to initial values"""
        self.truck_state = "NO_TRUCK"
        self.last_truck_position = None
        self.parking_timer_start = None

