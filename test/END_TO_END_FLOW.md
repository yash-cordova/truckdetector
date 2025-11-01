# End-to-End Video Processing Flow

## Complete Pipeline: Video → Detection → Monitoring → Display

### Architecture Overview
The application uses a **single-threaded processing model** with `SimpleVideoProcessor` that handles video reading, YOLO detection, and frame emission in one thread.

---

## Step-by-Step Flow

### 1. **User Action: Start Video**
```
User clicks "Start" button
    ↓
MainWindow.on_start()
    ↓
Creates SimpleVideoProcessor thread
```

**Code Location:** `window.py` → `on_start()`
```python
self.video_processor_thread = SimpleVideoProcessor(
    self.video_path,      # Video file path
    self.detector,        # YoloV5Detector instance
    self.video_canvas.roi, # ROI (not used anymore)
    start_frame=0,
    frame_skip_interval=self.frame_skip_interval  # From config.json
)
```

---

### 2. **SimpleVideoProcessor Thread Starts**
```
SimpleVideoProcessor.run() starts
    ↓
Opens video file with cv2.VideoCapture
    ↓
Sets buffer size to 1 (to avoid FFmpeg issues)
    ↓
Enters main loop: while self.running
```

**Code Location:** `threads.py` → `SimpleVideoProcessor.run()`

---

### 3. **Frame Reading Loop (Inside SimpleVideoProcessor)**
```
For each iteration:
    ↓
cv2.VideoCapture.read() → Reads ONE frame from video
    ↓
Frame format: numpy.ndarray (BGR format, shape: [height, width, 3])
    ↓
Check if frame_skip_count % frame_skip_interval == 0
    ↓
    ├─ YES → Run YOLO detection on this frame
    └─ NO  → Use cached detections from last YOLO run
```

**Key Logic:**
```python
# Every frame: Read from video
ret, frame = self.cap.read()

# Check if we should run YOLO detection
if self.frame_skip_count % self.frame_skip_interval == 0:
    # Run YOLO (expensive operation)
    detections = self.detector.detect(frame, roi=None)
    # Cache for next frames
    self.last_detections = detections
else:
    # Reuse cached detections (cheap)
    detections = self.last_detections

# Always emit frame + detections
self.frame_ready.emit(frame, detections, self.current_frame)
```

**Performance Optimization:**
- `frame_skip_interval = 1`: Run YOLO on every frame (slower, more accurate)
- `frame_skip_interval = 3`: Run YOLO every 3rd frame (faster, detections cached)
- Display always shows detections (either fresh or cached)

---

### 4. **YOLO Detection Process** (when frame_skip_interval triggers)
```
YoloV5Detector.detect(frame, roi=None)
    ↓
Input: frame (BGR numpy array)
    ↓
YOLOv5 model processes frame
    ↓
    - Resizes to inference_size (default: 640)
    - Runs inference on GPU/CPU
    - Applies confidence threshold (default: 0.35)
    - Applies IoU threshold (default: 0.45)
    ↓
Output: List of detections
```

**Detection Format:**
```python
detection = (x1, y1, x2, y2, confidence, class_id)
# Example: (100, 200, 300, 400, 0.85, 0)
#           ↑    ↑    ↑    ↑    ↑       ↑
#          left top right bot  conf   class
#          x1   y1   x2    y2          0=person, 1=forklift, 2=truck
```

**Code Location:** `detector.py` → `YoloV5Detector.detect()`

---

### 5. **Frame Emission (PyQt Signal)**
```
SimpleVideoProcessor emits signal:
    ↓
frame_ready.emit(frame, detections, frame_number)
    ↓
Signal connects to: MainWindow._on_frame_with_detections()
```

**Code Location:** `window.py` → Connection made in `on_start()`
```python
self.video_processor_thread.frame_ready.connect(self._on_frame_with_detections)
```

---

### 6. **MainWindow Receives Frame + Detections**
```
_on_frame_with_detections(frame, tracked_detections, frame_number)
    ↓
This method handles the complete processing pipeline:
```

**Step 6A: Update Timeline**
```python
# Calculate current time from frame number
current_time = frame_number / self.video_fps
# Update timeline slider position
```

**Step 6B: Process Truck Monitoring**
```python
_process_truck_monitoring(detections, frame)
    ↓
```

---

### 7. **Truck Monitoring Logic**
```
_process_truck_monitoring(detections, frame)
    ↓
Separate detections by class:
    - trucks = [d for d in detections if d[5] == 2]    # Class 2
    - persons = [d for d in detections if d[5] == 0]   # Class 0
    - forklifts = [d for d in detections if d[5] == 1] # Class 1
    ↓
STEP 1: Update Truck State
    ↓
TruckMonitor.update_truck_state(trucks, roi, parking_line_y)
    - Check if truck detected
    - Check if truck in ROI
    - Check if truck touched parking line
    - Update state: NO_TRUCK → TRUCK_DETECTED → TOUCHED_LINE → PARKED
    ↓
STEP 2: Check Person Violations
    ↓
TruckMonitor.check_person_violations(persons, trucks, roi, roi_corners)
    - Check if truck is moving (not PARKED or NO_TRUCK)
    - For each person: Check if bottom-center dot is in ROI
    - Returns: True if violation, False otherwise
    ↓
STEP 3: Check Persons/Forklifts in ROI
    ↓
TruckMonitor.check_person_or_forklift_in_roi(persons, forklifts, roi)
    - Returns: True if any person/forklift in ROI (for yellow light display)
    ↓
STEP 4: Update Status Display
    ↓
_update_status_display(has_violation, has_person_or_forklift_in_roi)
    - Get status info from TruckMonitor
    - Update traffic light (red/yellow/green)
    - Update status text
```

**Code Location:** 
- `window.py` → `_process_truck_monitoring()`
- `monitoring.py` → `TruckMonitor` class methods

---

### 8. **Draw Detections on Frame**
```
_draw_detections_with_ids(frame, detections)
    ↓
Calls: draw_detections_with_ids(frame, detections, model_names, roi)
    ↓
For each detection:
    - Check if detection touches ROI (filtering)
    - Draw bounding box (green rectangle)
    - Draw label (class name + ID + confidence)
    - Return annotated frame
```

**Detection Filtering:**
- Only draws detections that touch or overlap with ROI
- If detection is completely outside ROI, it's not drawn

**Code Location:** `detection_utils.py` → `draw_detections_with_ids()`

---

### 9. **Display Frame on Canvas**
```
video_canvas.set_frame(annotated_frame)
    ↓
VideoCanvas.paintEvent() is called
    ↓
    - Converts BGR → RGB
    - Creates QImage from frame
    - Scales to fit window (keeping aspect ratio)
    - Draws ROI polygon/rectangle overlay
    - Draws parking line overlay
    ↓
Frame displayed in UI
```

**Code Location:** `canvas.py` → `VideoCanvas.paintEvent()`

---

### 10. **Update UI Elements**
```
- Timeline slider: Updates position
- Time display: Updates current time
- Detection summary: Updates count and avg confidence
- Traffic light: Updates color (red/yellow/green)
- Status label: Updates text
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER ACTION                             │
│                    Click "Start" Button                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SimpleVideoProcessor                         │
│                         (QThread)                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Loop: while self.running                                │  │
│  │   1. cv2.VideoCapture.read() → frame (BGR numpy array)  │  │
│  │   2. Check: frame_skip_count % interval == 0?           │  │
│  │      ├─ YES → Run YOLO detection                         │  │
│  │      │         ↓                                           │  │
│  │      │      YoloV5Detector.detect(frame)                  │  │
│  │      │         ↓                                           │  │
│  │      │      Returns: [(x1,y1,x2,y2,conf,cls), ...]       │  │
│  │      │         ↓                                           │  │
│  │      │      Cache: self.last_detections = detections      │  │
│  │      │                                                    │  │
│  │      └─ NO  → Use cached: detections = last_detections    │  │
│  │                                                           │  │
│  │   3. Emit: frame_ready.emit(frame, detections, frame_num) │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ (PyQt Signal)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              MainWindow._on_frame_with_detections()             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. Update timeline slider position                       │  │
│  │  2. Process truck monitoring:                            │  │
│  │     - Separate detections (trucks, persons, forklifts)   │  │
│  │     - Update truck state (TruckMonitor)                  │  │
│  │     - Check violations (TruckMonitor)                     │  │
│  │     - Update status display                               │  │
│  │  3. Draw detections on frame:                             │  │
│  │     draw_detections_with_ids(frame, detections)          │  │
│  │  4. Display frame: video_canvas.set_frame(frame)          │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VideoCanvas Widget                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  paintEvent() called by Qt                                │  │
│  │    - Convert BGR → RGB                                    │  │
│  │    - Scale frame to fit window                            │  │
│  │    - Draw ROI overlay (green polygon/rectangle)          │  │
│  │    - Draw parking line overlay (red line)                  │  │
│  │    - Display frame in UI                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Data Structures

### 1. **Frame**
- **Type:** `numpy.ndarray`
- **Format:** BGR (Blue-Green-Red)
- **Shape:** `[height, width, 3]`
- **Example:** Frame from 1920x1080 video → Shape: `[1080, 1920, 3]`

### 2. **Detections List**
- **Type:** `List[Tuple]`
- **Format:** `[(x1, y1, x2, y2, confidence, class_id), ...]`
- **Example:** 
  ```python
  detections = [
      (100, 200, 300, 400, 0.85, 0),  # Person at (100,200) to (300,400), conf=0.85
      (500, 600, 800, 900, 0.92, 2),  # Truck at (500,600) to (800,900), conf=0.92
  ]
  ```

### 3. **Detection Fields**
- `x1, y1`: Top-left corner of bounding box
- `x2, y2`: Bottom-right corner of bounding box
- `confidence`: Detection confidence (0.0 to 1.0)
- `class_id`: 
  - `0` = person
  - `1` = forklift
  - `2` = truck

---

## Performance Optimizations

### 1. **Frame Skipping for YOLO**
- **Purpose:** YOLO detection is expensive (~50-200ms per frame)
- **Method:** Only run YOLO every Nth frame (`frame_skip_interval`)
- **Cache:** Reuse detections from last YOLO run for intermediate frames
- **Result:** Smooth video playback (always shows detections) + faster processing

**Example:**
```
frame_skip_interval = 3
Frame 0: Run YOLO → Cache detections
Frame 1: Use cached detections
Frame 2: Use cached detections
Frame 3: Run YOLO → Cache detections
Frame 4: Use cached detections
...
```

### 2. **Threading**
- **SimpleVideoProcessor** runs in separate thread (doesn't block UI)
- **PyQt Signals** used for thread-safe communication
- **UI updates** happen in main thread (Qt event loop)

### 3. **Video Buffer Settings**
- `CAP_PROP_BUFFERSIZE = 1`: Minimize buffering to avoid FFmpeg issues
- **Result:** More responsive frame reading, avoids async lock errors

---

## Timing Flow

```
Frame 0: [Read 20ms] → [YOLO 150ms] → [Processing 5ms] → [Display 10ms] = 185ms
Frame 1: [Read 20ms] → [Cache 0ms]  → [Processing 5ms] → [Display 10ms] = 35ms
Frame 2: [Read 20ms] → [Cache 0ms]  → [Processing 5ms] → [Display 10ms] = 35ms
Frame 3: [Read 20ms] → [YOLO 150ms] → [Processing 5ms] → [Display 10ms] = 185ms
...
Average: ~85ms per frame ≈ 11-12 FPS (with frame_skip_interval=3)
```

---

## Error Handling

1. **Video read failure:** Loop video back to start
2. **YOLO detection error:** Continue with empty detections
3. **Buffer full:** Skip frame (if skip_frames enabled) or wait
4. **Thread errors:** Graceful shutdown with cleanup

---

## Configuration

All settings in `config.json`:
- `frame_skip_interval`: How often to run YOLO (1=every frame, 3=every 3rd frame)
- `inference_size`: YOLO input size (640, 1280, etc.)
- `confidence_threshold`: Minimum confidence for detection (0.0-1.0)
- `iou_threshold`: NMS IoU threshold (0.0-1.0)
- `parking_duration_seconds`: How long truck must be still to be "PARKED"

---

This completes the end-to-end flow! Every frame goes through this pipeline from video file → detection → monitoring → display.

