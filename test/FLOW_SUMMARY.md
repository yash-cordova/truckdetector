# Video Processing Flow - Short Summary

## Quick Overview

**Single-threaded pipeline:** Video → YOLO Detection → Monitoring → Display

---

## Flow (5 Steps)

### 1. **Video Reading** (SimpleVideoProcessor Thread)
```
Opens video → Reads frame-by-frame in loop
```

### 2. **YOLO Detection** (Conditional)
```
Check: Is this the Nth frame? (frame_skip_interval)
    ├─ YES → Run YOLO → Get detections → Cache results
    └─ NO  → Use cached detections (skip expensive YOLO)
```

**Why skip frames?** YOLO is slow (~150ms). Running on every 3rd frame saves time while keeping detections visible.

### 3. **Emit Frame + Detections** (PyQt Signal)
```
Emit: (frame, detections, frame_number)
    ↓
MainWindow receives signal
```

### 4. **Process Monitoring** (MainWindow)
```
Separate detections:
    - Trucks (class 2)
    - Persons (class 0)  
    - Forklifts (class 1)
    ↓
Update truck state: NO_TRUCK → DETECTED → PARKED
    ↓
Check violations: Person bottom-center dot in ROI + Truck moving?
    ↓
Update status: Traffic light (red/yellow/green)
```

### 5. **Draw & Display**
```
Draw bounding boxes on frame
    ↓
Display frame in canvas (with ROI overlay)
```

---

## Key Points

**Performance:** Frame skipping reduces YOLO calls (every 3rd frame) while always showing detections (cached).

**No Buffers:** Simple architecture - frame read → process → emit → display (no shared buffers).

**Threading:** Only one thread (SimpleVideoProcessor) runs video processing. Main thread handles UI.

**Detection Format:** `(x1, y1, x2, y2, confidence, class_id)`

---

## Visual Flow

```
Video File
    ↓
[SimpleVideoProcessor Thread]
    ├─ Read Frame
    ├─ Check: Should run YOLO?
    │   ├─ YES → YOLO Detection → Cache
    │   └─ NO  → Use Cache
    └─ Emit Signal
    ↓
[Main Thread - MainWindow]
    ├─ Receive Frame + Detections
    ├─ Separate: Trucks, Persons, Forklifts
    ├─ Update Truck State
    ├─ Check Violations
    ├─ Draw Bounding Boxes
    └─ Display Frame
    ↓
Canvas Widget (UI)
```

---

## Timing Example

**With `frame_skip_interval = 3`:**
- Frame 0: Read → YOLO (150ms) → Process → Display = ~185ms
- Frame 1: Read → Cache (0ms) → Process → Display = ~35ms  
- Frame 2: Read → Cache (0ms) → Process → Display = ~35ms
- Frame 3: Read → YOLO (150ms) → Process → Display = ~185ms
- **Average:** ~85ms per frame ≈ **11-12 FPS**

---

That's it! Simple, single-threaded pipeline with frame-skipping optimization.

