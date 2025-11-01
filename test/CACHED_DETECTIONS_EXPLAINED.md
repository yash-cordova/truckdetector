# Cached Detections Explained

## What Does "Uses Cached Detections" Mean?

When YOLO is **NOT** run on a frame, the system reuses the **last detections** from when YOLO **was** run. This is called "caching."

---

## Simple Example

Imagine you have a video with frames: 0, 1, 2, 3, 4, 5, 6...

**With `frame_skip_interval = 3`:**

```
Frame 0: Run YOLO → Detects: 2 trucks, 1 person
         ↓
         Cache: self.last_detections = [truck1, truck2, person1]
         ↓
         Display frame with these detections

Frame 1: Skip YOLO → Use cache
         ↓
         detections = self.last_detections  (still: [truck1, truck2, person1])
         ↓
         Display frame with SAME detections as Frame 0

Frame 2: Skip YOLO → Use cache
         ↓
         detections = self.last_detections  (still: [truck1, truck2, person1])
         ↓
         Display frame with SAME detections as Frame 0

Frame 3: Run YOLO → Detects: 2 trucks, 0 persons (person left)
         ↓
         Cache: self.last_detections = [truck1, truck2]
         ↓
         Display frame with NEW detections

Frame 4: Skip YOLO → Use cache
         ↓
         detections = self.last_detections  (now: [truck1, truck2])
         ↓
         Display frame with SAME detections as Frame 3
```

---

## Code Explanation

### When YOLO Runs (Frame 0, 3, 6, 9...)
```python
if self.frame_skip_count % self.frame_skip_interval == 0:
    # Run YOLO detection (expensive, ~150ms)
    detections = self.detector.detect(frame, roi=None)
    # ↑ Fresh detections from current frame
    
    # SAVE detections to cache
    self.mutex.lock()
    self.last_detections = detections  # ← CACHE UPDATE
    self.mutex.unlock()
```

### When YOLO is Skipped (Frame 1, 2, 4, 5, 7, 8...)
```python
else:
    # Don't run YOLO (save time)
    # INSTEAD: Use the cached detections from last YOLO run
    self.mutex.lock()
    detections = self.last_detections  # ← USE CACHE
    self.mutex.unlock()
```

---

## Why Cache?

**Problem:** YOLO detection takes ~150ms per frame
- If you run YOLO on every frame (30 FPS), each frame takes ~150ms
- Video playback becomes VERY slow (only 6-7 FPS)

**Solution:** Cache detections
- Run YOLO every 3rd frame: 150ms + 150ms + 150ms = 450ms for 3 frames
- But use cached detections for other 2 frames: 0ms + 0ms = 0ms
- Average: 450ms / 3 frames = 150ms per frame (same YOLO cost)
- BUT: Video reading is faster (20ms vs 150ms), so overall: ~85ms per frame

**Result:** Faster playback (11-12 FPS) while still showing detections on every frame!

---

## Visual Comparison

### Without Caching (Run YOLO Every Frame)
```
Frame 0: [Read 20ms] → [YOLO 150ms] → Display = 170ms
Frame 1: [Read 20ms] → [YOLO 150ms] → Display = 170ms
Frame 2: [Read 20ms] → [YOLO 150ms] → Display = 170ms
Average: 170ms per frame = ~6 FPS ❌ SLOW
```

### With Caching (Run YOLO Every 3rd Frame)
```
Frame 0: [Read 20ms] → [YOLO 150ms] → Cache → Display = 170ms
Frame 1: [Read 20ms] → [Use Cache 0ms] → Display = 20ms  ⚡
Frame 2: [Read 20ms] → [Use Cache 0ms] → Display = 20ms  ⚡
Frame 3: [Read 20ms] → [YOLO 150ms] → Cache → Display = 170ms
Frame 4: [Read 20ms] → [Use Cache 0ms] → Display = 20ms  ⚡
Average: (170 + 20 + 20) / 3 = 70ms per frame = ~14 FPS ✅ FAST
```

---

## Important Points

1. **Cached detections are "old"** - They come from the last frame where YOLO ran
2. **Objects might move** - But detections stay the same until next YOLO run
3. **Trade-off:** Slight lag in detection updates, but MUCH faster playback
4. **Always shows detections** - Every frame has bounding boxes (either fresh or cached)

---

## Real Example

**Video shows a truck moving:**

- **Frame 0 (YOLO):** Truck at position (100, 200) → Cache this
- **Frame 1 (Cache):** Shows truck at (100, 200) - same as Frame 0 (truck actually moved to 105, 205, but we don't know)
- **Frame 2 (Cache):** Shows truck at (100, 200) - same as Frame 0 (truck actually moved to 110, 210)
- **Frame 3 (YOLO):** Truck at position (115, 215) → Update cache
- **Frame 4 (Cache):** Shows truck at (115, 215) - same as Frame 3

**Result:** Bounding boxes update every 3 frames, but video plays smoothly!

---

**Summary:** Caching = Reusing the last YOLO results to save time while keeping detections visible on every frame.

