# Code Refactoring Summary

## Structure Overview

The original `main.py` (2760 lines) has been refactored into logical modules for better organization and maintainability.

## File Structure

```
test/
├── main.py              # Original monolithic file (to be replaced)
├── main_new.py          # NEW: Entry point that imports everything
├── tracking.py          # KalmanFilter and Track classes
├── threads.py           # All thread classes (SimpleVideoProcessor, DetectionThread, etc.)
├── canvas.py            # VideoCanvas widget for displaying video frames
├── monitoring.py        # TruckMonitor class for truck state and violation logic
├── detection_utils.py  # Utility functions for drawing detections
└── window.py            # MainWindow class (TO BE CREATED)
```

## Module Responsibilities

### 1. `tracking.py`
- **KalmanFilter**: Enhanced Kalman filter for object tracking with motion prediction
- **Track**: Individual track object with state management

### 2. `threads.py`
- **SimpleVideoProcessor**: Main video processing thread (single-threaded architecture)
- **DetectionThread**: YOLO detection processing thread
- **FrameCollectorThread**: Frame collection and buffering thread

### 3. `canvas.py`
- **VideoCanvas**: PyQt5 widget for displaying video frames with ROI and parking line overlays

### 4. `monitoring.py`
- **TruckMonitor**: Handles truck state tracking, violation detection, and status management
  - `update_truck_state()`: Updates truck state based on detection
  - `check_person_violations()`: Checks for person violations
  - `check_person_or_forklift_in_roi()`: Checks if persons/forklifts are in ROI
  - `get_status_info()`: Returns status text and active light color

### 5. `detection_utils.py`
- **draw_detections_with_ids()**: Utility function to draw detections on frames with IDs and confidence scores

### 6. `window.py` (TO BE CREATED)
- **MainWindow**: Main PyQt5 application window
  - Video selection and playback controls
  - ROI configuration UI
  - Traffic light and status display
  - Fullscreen overlay
  - Timeline controls
  - Model loading and configuration

### 7. `main_new.py` (TO BE CREATED)
- Entry point that sets up environment variables and imports everything
- Initializes and runs the MainWindow

## Benefits of This Structure

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Easier Maintenance**: Changes to tracking logic don't affect UI code
3. **Better Testability**: Individual components can be tested in isolation
4. **Improved Readability**: Smaller, focused files are easier to understand
5. **Reusability**: Components can be reused in other projects

## Next Steps

1. ✅ Extract MainWindow class to `window.py` with proper imports - **COMPLETED**
2. ✅ Create `main_new.py` entry point - **COMPLETED**
3. ✅ Update imports in all modules to use relative/absolute imports correctly - **COMPLETED**
4. ⏳ Test the refactored structure - **PENDING**
5. ⏳ Remove or archive the original `main.py` once verified - **PENDING**

## File Organization

```
test/
├── main.py              # Original monolithic file (2760 lines) - KEEP AS BACKUP
├── main_new.py          # NEW: Entry point with environment setup
├── window.py            # NEW: MainWindow class (extracted from main.py)
├── tracking.py         # NEW: KalmanFilter and Track classes
├── threads.py           # NEW: All thread classes (SimpleVideoProcessor, etc.)
├── canvas.py            # NEW: VideoCanvas widget
├── monitoring.py        # NEW: TruckMonitor class for truck state and violations
├── detection_utils.py  # NEW: Utility functions for drawing detections
└── README_REFACTOR.md   # NEW: This documentation
```

## Notes

- The old monitoring methods (`_update_truck_state`, `_check_person_violations`, etc.) are still in `window.py` but are **not being called**. They can be removed later if desired, but are kept for reference.
- The `TruckMonitor` class is now used for all truck monitoring logic.
- All detection drawing logic has been moved to `detection_utils.py`.
- The `VideoCanvas` widget is now in its own module.

## Usage

Once complete, run the application using:
```bash
python test/main_new.py
```

Or if you rename `main_new.py` to `main.py`:
```bash
python test/main.py
```

