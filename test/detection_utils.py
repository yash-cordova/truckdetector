"""Utility functions for drawing detections and handling video operations"""
import cv2


def draw_detections_with_ids(frame, detections, model_names, roi=None):
    """Draw detections with unique IDs and confidence scores
    
    Only draws bounding boxes that touch or are inside the ROI.
    If object is partially in ROI, draws the FULL bounding box.
    
    Args:
        frame: Input frame (BGR)
        detections: List of detections in format (x1, y1, x2, y2, conf, class_id, obj_id?) or (x1, y1, x2, y2, conf, class_id)
        model_names: Dictionary mapping class_id to class name
        roi: ROI bounding box tuple (x1, y1, x2, y2) or None
        
    Returns:
        Frame with drawn detections
    """
    if not detections:
        return frame
        
    # Create a copy of the frame to draw on
    frame_copy = frame.copy()
    
    # Get ROI bounds for filtering
    roi_x1, roi_y1, roi_x2, roi_y2 = roi if roi else (0, 0, frame.shape[1], frame.shape[0])
    
    for detection in detections:
        if len(detection) >= 7:  # Has unique ID
            x1, y1, x2, y2, conf, class_id, obj_id = detection[:7]
        else:  # No unique ID (fallback)
            x1, y1, x2, y2, conf, class_id = detection[:6]
            obj_id = "N/A"
        
        # Check if detection touches or overlaps with ROI
        # Detection overlaps if: not completely outside ROI
        bbox_touches_roi = not (x2 < roi_x1 or x1 > roi_x2 or y2 < roi_y1 or y1 > roi_y2)
        
        # Only draw if touching ROI
        if not bbox_touches_roi:
            continue  # Skip this detection
        
        # Get class name
        class_name = model_names.get(class_id, f"Class_{class_id}")
        
        # Create label with format: "ObjectName ID:123 conf:0.85"
        if obj_id != "N/A":
            label = f"{class_name} ID:{obj_id} conf:{conf:.2f}"
        else:
            label = f"{class_name} conf:{conf:.2f}"
        
        # Draw FULL bounding box (even if partially outside ROI)
        cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame_copy, (int(x1), int(y1) - label_size[1] - 10), 
                    (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame_copy, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame_copy

