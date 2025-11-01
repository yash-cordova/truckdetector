import sys
import warnings
import os

# Fix FFmpeg async lock errors by setting environment before importing OpenCV
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_INTEL_MFX"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
os.environ["OPENCV_VIDEOIO_BACKEND"] = "CAP_ANY"
os.environ["OPENCV_THREADS"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QLabel,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QGroupBox,
    QFormLayout,
    QSizePolicy,
    QCheckBox,
    QSlider,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QWaitCondition
import cv2
import numpy as np
from detector import YoloV5Detector, CLASS_NAMES
import json
import queue
import time
import threading
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    """Enhanced Kalman Filter for object tracking with better motion prediction"""
    
    def __init__(self, x, y, w, h, class_id=0):
        # State vector: [x, y, w, h, vx, vy, vw, vh]
        self.state = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.class_id = class_id
        
        # State transition matrix with acceleration
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # w += vw
        self.F[3, 7] = 1  # h += vh
        
        # Measurement matrix (we observe x, y, w, h)
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1
        
        # Class-specific process noise
        if class_id == 0:  # person - more unpredictable
            self.Q = np.eye(8, dtype=np.float32) * 0.5
            self.R = np.eye(4, dtype=np.float32) * 2.0
        else:  # vehicles - more predictable
            self.Q = np.eye(8, dtype=np.float32) * 0.2
            self.R = np.eye(4, dtype=np.float32) * 1.0
        
        # Error covariance
        self.P = np.eye(8, dtype=np.float32) * 1000
        
        self.age = 0
        self.time_since_update = 0
        self.velocity_history = []  # Track velocity changes
        self.stationary_frames = 0  # Track consecutive stationary frames
        self.last_position = np.array([x, y], dtype=np.float32)
        
    def predict(self):
        """Predict next state with velocity decay for stationary objects"""
        # Check if object appears to be stationary
        current_position = self.state[:2]
        position_change = np.linalg.norm(current_position - self.last_position)
        
        if position_change < 2:  # Moving less than 2 pixels
            self.stationary_frames += 1
            # Decay velocity if stationary for multiple frames
            if self.stationary_frames > 3:
                # Strongly decay velocity towards zero
                self.state[4] *= 0.5  # vx
                self.state[5] *= 0.5  # vy
                self.state[6] *= 0.5  # vw
                self.state[7] *= 0.5  # vh
        else:
            self.stationary_frames = 0
        
        self.last_position = current_position.copy()
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        
    def update(self, measurement):
        """Update with measurement and track velocity changes"""
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        innovation = measurement - (self.H @ self.state)
        self.state = self.state + K @ innovation
        self.P = (np.eye(8) - K @ self.H) @ self.P
        
        # Track velocity changes for adaptive prediction
        current_velocity = self.state[4:6].copy()
        self.velocity_history.append(current_velocity)
        
        # Keep only recent velocity history
        if len(self.velocity_history) > 10:
            self.velocity_history = self.velocity_history[-10:]
        
        # Check for stationary state when getting measurement updates
        current_pos = self.state[:2]
        pos_change = np.linalg.norm(current_pos - self.last_position)
        
        if pos_change < 3:  # Object hasn't moved much
            # Reduce velocity estimate based on actual movement
            self.state[4] = current_pos[0] - self.last_position[0]  # vx = actual dx
            self.state[5] = current_pos[1] - self.last_position[1]  # vy = actual dy
        
        self.last_position = current_pos.copy()
        self.stationary_frames = 0  # Reset stationary counter on update
        self.time_since_update = 0
        
    def get_state(self):
        """Get current state as [x, y, w, h]"""
        return self.state[:4].copy()
        
    def get_velocity(self):
        """Get velocity as [vx, vy]"""
        return self.state[4:6].copy()


class Track:
    """Individual track object"""
    
    def __init__(self, track_id, detection, frame_id):
        x1, y1, x2, y2, conf, cls = detection
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        
        self.track_id = track_id
        self.class_id = cls
        self.confidence = conf
        self.kalman = KalmanFilter(x, y, w, h, cls)  # Pass class_id to Kalman filter
        self.last_detection = detection
        self.frame_id = frame_id
        self.age = 1
        self.time_since_update = 0
        self.state = 'Tentative'  # Tentative, Confirmed, Deleted
        self.missed_frames = 0  # Track consecutive missed frames
        
    def predict(self):
        """Predict next position"""
        self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection, frame_id):
        """Update track with new detection"""
        x1, y1, x2, y2, conf, cls = detection
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        measurement = np.array([x, y, w, h], dtype=np.float32)
        
        self.kalman.update(measurement)
        self.last_detection = detection
        self.frame_id = frame_id
        self.time_since_update = 0
        self.confidence = conf
        
        # Promote to confirmed after 3 updates
        if self.age >= 3:
            self.state = 'Confirmed'
            
    def get_bbox(self):
        """Get bounding box as [x1, y1, x2, y2]"""
        x, y, w, h = self.kalman.get_state()
        return [x, y, x + w, y + h]
        
    def get_center(self):
        """Get center point"""
        x, y, w, h = self.kalman.get_state()
        return [x + w/2, y + h/2]


# Removed OCSortTracker - no tracking needed

class OCSortTracker_REMOVED:
    """OC-SORT (Object-Centric SORT) tracker with threading support"""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.2, 
                 distance_threshold=100, velocity_threshold=50):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.velocity_threshold = velocity_threshold
        
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        
        # Class-specific thresholds for better tracking
        self.class_thresholds = {
            0: {"distance": 80, "velocity": 40, "iou": 0.15},  # person - more flexible
            1: {"distance": 60, "velocity": 30, "iou": 0.25},  # forklift
            2: {"distance": 50, "velocity": 20, "iou": 0.3}    # truck - more strict
        }
        
        # Thread safety
        self.mutex = threading.Lock()
        
    def update(self, detections, frame_id):
        """Update tracker with new detections"""
        with self.mutex:
            self.frame_count = frame_id
            
            # Predict all tracks
            for track in self.tracks.values():
                track.predict()
                
            # Associate detections with tracks
            matched_tracks, unmatched_detections, unmatched_tracks = self._associate_detections(
                detections, self.tracks
            )
            
            # Update matched tracks
            for track_id, detection_idx in matched_tracks:
                track = self.tracks[track_id]
                track.update(detections[detection_idx], frame_id)
                
            # Create new tracks for unmatched detections
            for detection_idx in unmatched_detections:
                track_id = self.next_id
                self.next_id += 1
                track = Track(track_id, detections[detection_idx], frame_id)
                self.tracks[track_id] = track
                print(f"OC-SORT: Created new track {track_id} for detection {detection_idx} (class {detections[detection_idx][5]})")
                
            # Remove old tracks
            tracks_to_remove = []
            for track_id, track in self.tracks.items():
                if track.time_since_update > self.max_age:
                    tracks_to_remove.append(track_id)
                    print(f"OC-SORT: Removing old track {track_id} (class {track.class_id}) after {track.time_since_update} frames")
                    
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
                
            # Return confirmed tracks with IDs
            confirmed_tracks = []
            for track in self.tracks.values():
                if track.state == 'Confirmed':
                    bbox = track.get_bbox()
                    track_data = (
                        bbox[0], bbox[1], bbox[2], bbox[3],  # x1, y1, x2, y2
                        track.confidence,                    # confidence
                        track.class_id,                      # class
                        track.track_id                       # track ID
                    )
                    confirmed_tracks.append(track_data)
                    
            return confirmed_tracks
            
    def _associate_detections(self, detections, tracks):
        """Associate detections with tracks using Hungarian algorithm"""
        if not detections or not tracks:
            return [], list(range(len(detections))), list(tracks.keys())
            
        # Calculate cost matrix
        cost_matrix = self._calculate_cost_matrix(detections, tracks)
        
        # Use Hungarian algorithm for optimal assignment
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Filter assignments based on cost threshold
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(tracks.keys())
        
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            # Get class-specific threshold
            track_id = list(tracks.keys())[track_idx]
            track = tracks[track_id]
            detection = detections[detection_idx]
            
            # Use class-specific thresholds
            class_id = track.class_id
            thresholds = self.class_thresholds.get(class_id, {
                "distance": self.distance_threshold, 
                "velocity": self.velocity_threshold, 
                "iou": self.iou_threshold
            })
            
            if cost_matrix[track_idx, detection_idx] < thresholds["distance"]:
                matched_tracks.append((track_id, detection_idx))
                unmatched_detections.remove(detection_idx)
                unmatched_tracks.remove(track_id)
                print(f"OC-SORT: Matched track {track_id} (class {class_id}) with detection {detection_idx}, cost: {cost_matrix[track_idx, detection_idx]:.2f}")
            else:
                print(f"OC-SORT: Rejected match track {track_id} (class {class_id}) with detection {detection_idx}, cost: {cost_matrix[track_idx, detection_idx]:.2f} > {thresholds['distance']}")
                
        return matched_tracks, unmatched_detections, unmatched_tracks
        
    def _calculate_cost_matrix(self, detections, tracks):
        """Calculate cost matrix between detections and tracks"""
        cost_matrix = np.full((len(tracks), len(detections)), float('inf'))
        
        for i, (track_id, track) in enumerate(tracks.items()):
            track_bbox = track.get_bbox()
            track_center = track.get_center()
            
            # Get class-specific thresholds for this track
            class_id = track.class_id
            thresholds = self.class_thresholds.get(class_id, {
                "distance": self.distance_threshold, 
                "velocity": self.velocity_threshold, 
                "iou": self.iou_threshold
            })
            
            for j, detection in enumerate(detections):
                x1, y1, x2, y2, conf, cls = detection
                detection_center = [(x1 + x2) / 2, (y1 + y2) / 2]
                
                # Calculate IoU
                iou = self._calculate_iou(track_bbox, [x1, y1, x2, y2])
                
                # Calculate distance
                distance = np.sqrt(
                    (track_center[0] - detection_center[0])**2 + 
                    (track_center[1] - detection_center[1])**2
                )
                
                # Calculate velocity consistency
                velocity_cost = self._calculate_velocity_cost(track, detection)
                
                # Combined cost (lower is better) with class-specific thresholds
                if iou > thresholds["iou"]:
                    # Weighted cost based on class type
                    if class_id == 0:  # person - prioritize distance over velocity
                        cost = distance * 0.7 + velocity_cost * 0.3
                    else:  # vehicles - more balanced
                        cost = distance * 0.5 + velocity_cost * 0.5
                    
                    # Penalize high costs for stationary objects to prevent ID switching
                    if track.kalman.stationary_frames > 5:
                        # If object has been stationary, apply additional penalty for distant detections
                        # This prevents the track from "jumping" to a new detection
                        cost = cost * (1 + track.kalman.stationary_frames * 0.1)
                else:
                    cost = float('inf')
                    
                cost_matrix[i, j] = cost
                
        return cost_matrix
        
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_velocity_cost(self, track, detection):
        """Calculate velocity consistency cost with class-specific handling"""
        if track.age < 2:
            return 0
            
        # Get predicted position
        predicted_bbox = track.get_bbox()
        predicted_center = [(predicted_bbox[0] + predicted_bbox[2]) / 2,
                           (predicted_bbox[1] + predicted_bbox[3]) / 2]
        
        # Get detection center
        x1, y1, x2, y2, conf, cls = detection
        detection_center = [(x1 + x2) / 2, (y1 + y2) / 2]
        
        # Calculate velocity difference
        velocity_diff = np.sqrt(
            (predicted_center[0] - detection_center[0])**2 + 
            (predicted_center[1] - detection_center[1])**2
        )
        
        # Get class-specific velocity threshold
        class_id = track.class_id
        thresholds = self.class_thresholds.get(class_id, {
            "velocity": self.velocity_threshold
        })
        
        # Apply class-specific velocity penalty
        if class_id == 0:  # person - more forgiving for direction changes
            return min(velocity_diff * 0.5, thresholds["velocity"])
        else:  # vehicles - more strict
            return min(velocity_diff, thresholds["velocity"])


class DetectionThread(QThread):
    """Separate thread for running YOLO detection"""
    detection_complete = pyqtSignal(np.ndarray, list, int)  # frame, detections, frame_number
    
    def __init__(self, detector, roi=None):
        super().__init__()
        self.detector = detector
        self.roi = roi
        self.running = False
        self.mutex = QMutex()
        self.detection_queue = queue.Queue(maxsize=30)  # Larger queue
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    def add_frame(self, frame, frame_number):
        """Add frame for detection processing"""
        try:
            self.detection_queue.put_nowait((frame, frame_number))
        except queue.Full:
            # Remove oldest frame to make room
            try:
                self.detection_queue.get_nowait()
                self.detection_queue.put_nowait((frame, frame_number))
            except:
                pass  # Skip frame if still failing
            
    def set_roi(self, roi):
        self.mutex.lock()
        self.roi = roi
        self.mutex.unlock()
        
    def run(self):
        self.running = True
        print("DetectionThread: Starting detection processing")
        
        while self.running:
            try:
                # Get frame with timeout
                try:
                    frame, frame_number = self.detection_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process detection
                detections = []
                try:
                    if self.detector:
                        self.mutex.lock()
                        roi = self.roi
                        self.mutex.unlock()
                        
                        # Run detection with timeout protection
                        try:
                            detections = self.detector.detect(frame, roi)
                        except Exception as e:
                            print(f"YOLO detection error: {e}")
                            detections = []
                    
                    # Track performance
                    self.frame_count += 1
                    current_time = time.time()
                    if self.frame_count % 100 == 0:
                        elapsed = current_time - self.last_frame_time
                        fps = 100 / elapsed if elapsed > 0 else 0
                        print(f"DetectionThread: Processed {self.frame_count} frames, avg FPS: {fps:.2f}")
                        self.last_frame_time = current_time
                    
                    # Emit results
                    self.detection_complete.emit(frame, detections or [], frame_number)
                    
                except Exception as e:
                    print(f"Detection processing error: {e}")
                    try:
                        self.detection_complete.emit(frame, [], frame_number)
                    except:
                        pass
                
                try:
                    self.detection_queue.task_done()
                except:
                    pass
                
            except Exception as e:
                print(f"DetectionThread error: {e}")
                import traceback
                traceback.print_exc()
                
        print("DetectionThread: Finished")
        
    def stop(self):
        self.running = False


class TrackingThread(QThread):
    """Separate thread for OC-SORT tracking"""
    tracking_complete = pyqtSignal(np.ndarray, list, int)  # frame, tracked_detections, frame_number
    
    def __init__(self, ocsort_tracker):
        super().__init__()
        self.ocsort_tracker = ocsort_tracker
        self.running = False
        self.tracking_queue = queue.Queue(maxsize=10)
        
    def add_detection(self, frame, detections, frame_number):
        """Add detection for tracking processing"""
        try:
            self.tracking_queue.put_nowait((frame, detections, frame_number))
        except queue.Full:
            # Skip if queue is full
            pass
            
    def run(self):
        self.running = True
        print("TrackingThread: Starting tracking processing")
        
        while self.running:
            try:
                frame, detections, frame_number = self.tracking_queue.get(timeout=0.1)
                
                # Run OC-SORT tracking
                tracked_detections = self.ocsort_tracker.update(detections, frame_number)
                self.tracking_complete.emit(frame, tracked_detections, frame_number)
                
                self.tracking_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TrackingThread error: {e}")
                
        print("TrackingThread: Finished")
        
    def stop(self):
        self.running = False


class SimpleVideoProcessor(QThread):
    """Simple single-threaded video processor - no complex queues"""
    frame_ready = pyqtSignal(np.ndarray, list, int)  # frame, tracked_detections, frame_number
    
    def __init__(self, video_path, detector, roi=None, start_frame=0, frame_skip_interval=1):
        super().__init__()
        self.video_path = video_path
        self.detector = detector
        self.roi = roi  # Not used anymore but kept for compatibility
        self.running = False
        self.cap = None
        self.start_frame = start_frame
        self.current_frame = 0
        self.last_detections = []  # Cache last detections (fallback only)
        self.mutex = QMutex()  # Thread-safe access to last_detections
        self.frame_skip_interval = frame_skip_interval  # Frame skip interval from config
        self.frame_skip_count = 0  # Counter for frame skipping
        
    def run(self):
        self.running = True
        print("SimpleVideoProcessor: Starting")
        
        try:
            # Open video
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print("Failed to open video")
                return
            
            # Set buffer size to 1 to avoid FFmpeg issues
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Seek to start frame if specified
            if self.start_frame > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                self.current_frame = self.start_frame
                print(f"Seeking to frame {self.start_frame}")
            
            while self.running:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame = 0
                        continue
                    
                    # Run detection based on frame_skip_interval from config
                    # This allows skipping frames for performance while maintaining smooth display
                    if self.detector:
                        if self.frame_skip_count % self.frame_skip_interval == 0:
                            # Run detection on selected frames (based on config)
                            detections = self.detector.detect(frame, roi=None)
                            
                            # Cache detections for skipped frames
                            self.mutex.lock()
                            self.last_detections = detections
                            self.mutex.unlock()
                        else:
                            # Use cached detections for skipped frames
                            self.mutex.lock()
                            detections = self.last_detections
                            self.mutex.unlock()
                    else:
                        detections = []
                    
                    # Emit results (always emit frame, with fresh or cached detections)
                    self.frame_ready.emit(frame, detections, self.current_frame)
                    
                    # Increment frame skip counter
                    self.frame_skip_count += 1
                    
                    self.current_frame += 1
                    
                    # Small delay to control speed
                    self.msleep(20)  # ~50 FPS display (allows fast playback)
                    
                except Exception as e:
                    print(f"SimpleVideoProcessor frame error: {e}")
                    if not self.cap or not self.cap.isOpened():
                        break
                    self.current_frame += 1
                    
        except Exception as e:
            print(f"SimpleVideoProcessor error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()
                print("SimpleVideoProcessor: Released video capture")
        
        print("SimpleVideoProcessor: Finished")
    
    def stop(self):
        self.running = False


class YoloProcessorThread(QThread):
    """Thread for YOLO detection processing"""
    detection_ready = pyqtSignal(np.ndarray, list, int)  # frame, detections, frame_number
    
    def __init__(self, detector, shared_buffer, roi=None, use_ocsort=False):
        super().__init__()
        self.detector = detector
        self.shared_buffer = shared_buffer
        self.roi = roi
        self.use_ocsort = use_ocsort
        self.running = False
        self.mutex = QMutex()
        self.roi_mutex = QMutex()
        
    def set_roi(self, roi):
        self.roi_mutex.lock()
        self.roi = roi
        self.roi_mutex.unlock()
        
    def run(self):
        self.running = True
        print("YoloProcessorThread: Starting")
        
        while self.running:
            try:
                # Get frame from buffer with timeout
                try:
                    frame, frame_number = self.shared_buffer.get(timeout=0.1)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Buffer get error: {e}")
                    continue
                
                # Run detection if detector available
                detections = []
                if self.detector:
                    try:
                        self.roi_mutex.lock()
                        roi = self.roi
                        self.roi_mutex.unlock()
                        
                        detections = self.detector.detect(frame, roi)
                    except Exception as e:
                        print(f"Detection error: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Emit detection results
                try:
                    self.detection_ready.emit(frame, detections or [], frame_number)
                except Exception as e:
                    print(f"Signal emit error: {e}")
                
                try:
                    self.shared_buffer.task_done()
                except:
                    pass  # Ignore task_done errors
                
            except Exception as e:
                print(f"YoloProcessorThread error: {e}")
                import traceback
                traceback.print_exc()
                
        print("YoloProcessorThread: Finished")
        
    def stop(self):
        self.running = False


class FrameCollectorThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, int)  # frame, frame_number
    
    def __init__(self, video_path, shared_buffer, buffer_size=10):
        super().__init__()
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.shared_buffer = shared_buffer
        self.cap = None
        self.running = False
        self.skip_frames = False
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.start_frame = 0  # Frame to start from
        self.max_failures = 3  # Track consecutive read failures
        self.failure_count = 0
        
    def set_skip_frames(self, skip):
        self.mutex.lock()
        self.skip_frames = skip
        self.mutex.unlock()
        
        
    def run(self):
        try:
            if not self.video_path:
                print("FrameCollector: No video path provided")
                return
                
            print(f"FrameCollector: Opening video: {self.video_path}")
            # Open with specific flags to avoid FFmpeg threading issues
            self.cap = cv2.VideoCapture(self.video_path)
            
            # Try to set backend explicitly to avoid threading issues
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass
                
            if not self.cap.isOpened():
                print("FrameCollector: Failed to open video")
                return
                
            self.running = True
            frame_number = self.start_frame
            print(f"FrameCollector: Starting frame collection from frame {frame_number}")
            
            # Set starting position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            
            while self.running:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        # End of video - loop back to start
                        print(f"FrameCollector: End of video at frame {frame_number}, looping back to start")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_number = 0
                        self.failure_count = 0
                        
                        # Try to read first frame after loop
                        ret, frame = self.cap.read()
                        if not ret:
                            self.failure_count += 1
                            print(f"FrameCollector: Could not read after loop (failure {self.failure_count}/{self.max_failures})")
                            if self.failure_count >= self.max_failures:
                                print("FrameCollector: Too many read failures after loop")
                                break
                        continue
                        
                    self.failure_count = 0  # Reset on successful read
                    
                    self.mutex.lock()
                    try:
                        if self.skip_frames:
                            # Skip frame if buffer is full
                            if self.shared_buffer.full():
                                frame_number += 1
                                continue
                        else:
                            # Wait if buffer is full
                            wait_count = 0
                            while self.shared_buffer.full() and self.running and wait_count < 100:
                                self.condition.wait(self.mutex, 10)
                                wait_count += 1
                                
                        if self.running and not self.shared_buffer.full():
                            try:
                                # Try to put frame in buffer
                                self.shared_buffer.put_nowait((frame.copy(), frame_number))
                                self.frame_ready.emit(frame, frame_number)
                                frame_number += 1
                            except queue.Full:
                                # Buffer became full while we checked
                                pass
                            except Exception as e:
                                print(f"Error putting frame in buffer: {e}")
                    finally:
                        self.mutex.unlock()
                except Exception as e:
                    print(f"FrameCollector frame reading error: {e}")
                    import traceback
                    traceback.print_exc()
                    self.failure_count += 1
                    if self.failure_count >= self.max_failures:
                        break
                    
            print("FrameCollector: Thread finished")
        except Exception as e:
            print(f"FrameCollector error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                if self.cap:
                    self.cap.release()
                    print("FrameCollector: Released video capture")
            except Exception as e:
                print(f"Error releasing capture: {e}")
                
    def stop(self):
        self.running = False
        try:
            self.condition.wakeAll()
        except:
            pass
        try:
            if self.cap:
                self.cap.release()
        except:
            pass




class VideoCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._frame = None
        self.roi = None  # (x1, y1, x2, y2) - bounding box for compatibility
        self.roi_corners = None  # List of 4 corners for polygon

    def set_frame(self, frame_bgr):
        self._frame = frame_bgr
        self.update()

    def set_roi(self, roi):
        self.roi = roi
        self.update()

    def set_roi_corners(self, corners):
        self.roi_corners = corners
        self.update()
        
    def set_parking_line_points(self, points):
        self.parking_line_points = points
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._frame is None:
            return

        frame = self._frame
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        painter = QtGui.QPainter(self)
        try:
            # scale to fit
            target = self.rect()
            scaled = pix.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (target.width() - scaled.width()) // 2
            y = (target.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # Draw ROI polygon if corners are available
            if self.roi_corners is not None and len(self.roi_corners) >= 3:
                scale = min(target.width() / w, target.height() / h)
                points = []
                for corner_x, corner_y in self.roi_corners:
                    sx = x + int(corner_x * scale)
                    sy = y + int(corner_y * scale)
                    points.append(QtCore.QPoint(sx, sy))
                
                pen = QtGui.QPen(QtGui.QColor(0, 255, 0), 2)
                painter.setPen(pen)
                painter.drawPolygon(points)
            elif self.roi is not None:
                # Fallback to rectangle
                x1, y1, x2, y2 = self.roi
                scale = min(target.width() / w, target.height() / h)
                sx = x + int(x1 * scale)
                sy = y + int(y1 * scale)
                ex = x + int(x2 * scale)
                ey = y + int(y2 * scale)
                pen = QtGui.QPen(QtGui.QColor(0, 255, 0), 2)
                painter.setPen(pen)
                painter.drawRect(QtCore.QRect(QtCore.QPoint(sx, sy), QtCore.QPoint(ex, ey)))
                
            # Draw parking line if points are available
            if hasattr(self, 'parking_line_points') and self.parking_line_points is not None and len(self.parking_line_points) == 2:
                scale = min(target.width() / w, target.height() / h)
                p1_x, p1_y = self.parking_line_points[0]
                p2_x, p2_y = self.parking_line_points[1]
                sp1_x = x + int(p1_x * scale)
                sp1_y = y + int(p1_y * scale)
                sp2_x = x + int(p2_x * scale)
                sp2_y = y + int(p2_y * scale)
                
                pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 3)  # Red line
                painter.setPen(pen)
                painter.drawLine(sp1_x, sp1_y, sp2_x, sp2_y)
        finally:
            painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Watchman - Dock Monitoring")
        self.resize(1280, 720)

        # State
        self.video_path = None
        self.fullscreen = False
        self.detector = None
        self.model_names = CLASS_NAMES
        
        # Simplified architecture - single thread processing
        self.video_processor_thread = None
        self.last_frame_received = time.time()
        
        # No tracking - just use detections as-is
        
        # Video capture for seeking
        self.seek_cap = None
        
        # Timeline state management
        self.is_playing = False
        self._timeline_dragging = False
        
        # Video properties for time-based navigation
        self.video_fps = 30.0  # Default FPS
        self.video_duration = 0.0  # Duration in seconds
        self.current_time = 0.0  # Current time in seconds
        self.frame_skip_count = 0  # Frame skip counter for YOLO
        self.frame_skip_interval = 1  # Process every Nth frame (1 = every frame, 2 = every other frame, etc.)
        
        # Non-blocking seeking
        self.seek_timer = QTimer()
        self.seek_timer.timeout.connect(self._process_seek_queue)
        self.seek_timer.setSingleShot(True)
        self.pending_seek_time = None
        self.seek_in_progress = False
        self.operation_lock = False
        self.operation_mutex = QMutex()  # Add mutex for operation lock
        self.scroll_debounce_timer = QTimer()  # Add debounce timer for rapid scrolling
        self.scroll_debounce_timer.setSingleShot(True)
        self.scroll_debounce_timer.timeout.connect(self._process_debounced_scroll)
        
        # Add timer to print debug info every second
        self.debug_timer = QTimer()
        self.debug_timer.timeout.connect(self._print_debug_info)
        self.debug_timer.start(1000)  # Every 1 second
        self.last_debug_data = {}  # Store last debug data
        
        # Truck monitoring state
        self.truck_state = "NO_TRUCK"  # NO_TRUCK, TRUCK_DETECTED, TOUCHED_LINE, PARKED
        self.last_truck_position = None
        self.parking_line_y = None  # Will be set based on ROI
        self.parking_line_points = None  # Will be set based on user input
        self.parking_timer_start = None  # Time when truck touched parking line
        self.parking_duration_seconds = None  # Will be loaded from config.json
        
        # OC-SORT tracking configuration
        self.use_ocsort_tracking = True  # Enable OC-SORT tracking
        self.ocsort_max_age = 30  # Maximum frames to keep lost tracks
        self.ocsort_min_hits = 3  # Minimum hits to confirm track
        self.ocsort_iou_threshold = 0.3  # IoU threshold for association
        self.ocsort_distance_threshold = 50  # Distance threshold for association
        self.ocsort_velocity_threshold = 10  # Velocity threshold for association
        
        # Configuration
        self.detection_confidence_min = 0.3  # minimum confidence for detection

        # Right: video canvas
        self.video_canvas = VideoCanvas()

        # Left: controls
        self.btn_select = QPushButton("Select Video")
        self.btn_select.clicked.connect(self.on_select_video)

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.on_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.on_stop)

        self.btn_full = QPushButton("Fullscreen Video")
        self.btn_full.clicked.connect(self.on_toggle_fullscreen)
        
        # Skip frames checkbox
        self.skip_frames_cb = QCheckBox("Skip frames when buffer full")
        self.skip_frames_cb.setChecked(True)

        # ROI inputs - 10 boxes for 5 corners (x,y for each corner)
        self.input_corner1_x = QLineEdit("900")
        self.input_corner1_y = QLineEdit("650")
        self.input_corner2_x = QLineEdit("1750")
        self.input_corner2_y = QLineEdit("700")
        self.input_corner3_x = QLineEdit("1250")
        self.input_corner3_y = QLineEdit("1950")
        self.input_corner4_x = QLineEdit("0")
        self.input_corner4_y = QLineEdit("1950")
        self.input_corner5_x = QLineEdit("0")
        self.input_corner5_y = QLineEdit("1200")
        
        # Parking line inputs - 2 points to define the parking line
        self.input_parking_line1_x = QLineEdit("500")
        self.input_parking_line1_y = QLineEdit("900")
        self.input_parking_line2_x = QLineEdit("1600")
        self.input_parking_line2_y = QLineEdit("1100")
        
        roi_inputs = [self.input_corner1_x, self.input_corner1_y, self.input_corner2_x, self.input_corner2_y,
                     self.input_corner3_x, self.input_corner3_y, self.input_corner4_x, self.input_corner4_y,
                     self.input_corner5_x, self.input_corner5_y]
        
        parking_line_inputs = [self.input_parking_line1_x, self.input_parking_line1_y, 
                              self.input_parking_line2_x, self.input_parking_line2_y]
        
        for le in roi_inputs:
            le.setPlaceholderText("int")
            le.setMaximumWidth(80)
            le.editingFinished.connect(self.on_roi_changed)
            
        for le in parking_line_inputs:
            le.setPlaceholderText("int")
            le.setMaximumWidth(80)
            le.editingFinished.connect(self.on_roi_changed)

        form = QFormLayout()
        form.addRow("Corner 1 (x,y):", QWidget())  # Header
        corner1_layout = QHBoxLayout()
        corner1_layout.addWidget(QLabel("x:"))
        corner1_layout.addWidget(self.input_corner1_x)
        corner1_layout.addWidget(QLabel("y:"))
        corner1_layout.addWidget(self.input_corner1_y)
        form.addRow("", corner1_layout)
        
        form.addRow("Corner 2 (x,y):", QWidget())  # Header
        corner2_layout = QHBoxLayout()
        corner2_layout.addWidget(QLabel("x:"))
        corner2_layout.addWidget(self.input_corner2_x)
        corner2_layout.addWidget(QLabel("y:"))
        corner2_layout.addWidget(self.input_corner2_y)
        form.addRow("", corner2_layout)
        
        form.addRow("Corner 3 (x,y):", QWidget())  # Header
        corner3_layout = QHBoxLayout()
        corner3_layout.addWidget(QLabel("x:"))
        corner3_layout.addWidget(self.input_corner3_x)
        corner3_layout.addWidget(QLabel("y:"))
        corner3_layout.addWidget(self.input_corner3_y)
        form.addRow("", corner3_layout)
        
        form.addRow("Corner 4 (x,y):", QWidget())  # Header
        corner4_layout = QHBoxLayout()
        corner4_layout.addWidget(QLabel("x:"))
        corner4_layout.addWidget(self.input_corner4_x)
        corner4_layout.addWidget(QLabel("y:"))
        corner4_layout.addWidget(self.input_corner4_y)
        form.addRow("", corner4_layout)
        
        form.addRow("Corner 5 (x,y):", QWidget())  # Header
        corner5_layout = QHBoxLayout()
        corner5_layout.addWidget(QLabel("x:"))
        corner5_layout.addWidget(self.input_corner5_x)
        corner5_layout.addWidget(QLabel("y:"))
        corner5_layout.addWidget(self.input_corner5_y)
        form.addRow("", corner5_layout)
        
        # Parking line inputs
        form.addRow("Parking Line Point 1 (x,y):", QWidget())  # Header
        parking_line1_layout = QHBoxLayout()
        parking_line1_layout.addWidget(QLabel("x:"))
        parking_line1_layout.addWidget(self.input_parking_line1_x)
        parking_line1_layout.addWidget(QLabel("y:"))
        parking_line1_layout.addWidget(self.input_parking_line1_y)
        form.addRow("", parking_line1_layout)
        
        form.addRow("Parking Line Point 2 (x,y):", QWidget())  # Header
        parking_line2_layout = QHBoxLayout()
        parking_line2_layout.addWidget(QLabel("x:"))
        parking_line2_layout.addWidget(self.input_parking_line2_x)
        parking_line2_layout.addWidget(QLabel("y:"))
        parking_line2_layout.addWidget(self.input_parking_line2_y)
        form.addRow("", parking_line2_layout)

        roi_group = QGroupBox("ROI Settings")
        roi_group.setLayout(form)

        # Model info (hardcoded)
        self.model_label = QLabel("Model: YOLOv5")
        # Dynamic class filters area (populated after model loads)
        self.class_area = QtWidgets.QScrollArea()
        self.class_area.setWidgetResizable(True)
        self.class_container = QWidget()
        self.class_layout = QVBoxLayout()
        self.class_layout.setContentsMargins(6, 6, 6, 6)
        self.class_container.setLayout(self.class_layout)
        self.class_area.setWidget(self.class_container)
        self.class_checkboxes = []

        # Control layout
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.btn_select)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addWidget(self.btn_full)
        controls_layout.addWidget(self.skip_frames_cb)
        
        controls_layout.addWidget(roi_group)
        # Detection summary
        self.summary_label = QLabel("Detections: 0 | Avg conf: -")
        controls_layout.addWidget(self.model_label)
        controls_layout.addWidget(QLabel("Classes to display"))
        controls_layout.addWidget(self.class_area)
        controls_layout.addWidget(self.summary_label)
        # Video timeline slider (time-based)
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setTracking(False)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1000)  # 1000 steps for precise time control
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderPressed.connect(self._on_timeline_pressed)
        self.timeline_slider.sliderReleased.connect(self._on_timeline_released)
        self.timeline_slider.valueChanged.connect(self._on_timeline_changed)
        
        # Timeline controls
        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(QLabel("Timeline:"))
        timeline_layout.addWidget(self.timeline_slider)
        controls_layout.addLayout(timeline_layout)
        
        # Time information
        self.time_info_label = QLabel("Time: 00:00 / 00:00")
        controls_layout.addWidget(self.time_info_label)
        
        # Playback status indicator
        self.playback_status_label = QLabel("Status: Stopped")
        self.playback_status_label.setAlignment(Qt.AlignCenter)
        self.playback_status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: #fff;
            }
        """)
        controls_layout.addWidget(self.playback_status_label)
        
        # Traffic Light and Status Section
        traffic_group = QGroupBox("Traffic Status")
        traffic_layout = QVBoxLayout()
        
        # Traffic Light Widget - 3 horizontal lights
        self.traffic_light_widget = QWidget()
        self.traffic_light_widget.setFixedSize(200, 80)
        self.traffic_light_widget.setStyleSheet("""
            QWidget {
                background-color: #2c2c2c;
                border: 3px solid #1a1a1a;
                border-radius: 15px;
            }
        """)
        
        # Create three light labels
        self.red_light = QLabel("●")
        self.yellow_light = QLabel("●")
        self.green_light = QLabel("●")
        
        # Set up layout for the traffic light (horizontal)
        light_layout = QHBoxLayout()
        light_layout.setSpacing(10)
        light_layout.setContentsMargins(15, 10, 15, 10)
        
        # Style the lights
        for light in [self.red_light, self.yellow_light, self.green_light]:
            light.setAlignment(Qt.AlignCenter)
            light.setFixedSize(50, 50)
            light.setStyleSheet("""
                QLabel {
                    font-size: 30px;
                    border-radius: 25px;
                    background-color: #404040;
                    color: #666666;
                }
            """)
            light_layout.addWidget(light)
        
        self.traffic_light_widget.setLayout(light_layout)
        traffic_layout.addWidget(self.traffic_light_widget)
        
        
        # Initialize traffic light to green (no truck state)
        self._set_traffic_light("green")
        
        # Status Label
        self.status_label = QLabel("Status: NO_TRUCK")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
        """)
        traffic_layout.addWidget(self.status_label)
        
        
        traffic_group.setLayout(traffic_layout)
        controls_layout.addWidget(traffic_group)
        
        controls_layout.addStretch(1)

        controls = QWidget()
        controls.setLayout(controls_layout)
        controls.setMinimumWidth(260)
        controls.setStyleSheet("""
            QGroupBox{font-weight:bold;} 
            QPushButton{height:30px;}
            QPushButton:hover{background-color: #e0e0e0;}
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #d4d4d4, stop:1 #afafaf);
            }
        """)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self.video_canvas)
        splitter.setStretchFactor(0, 1)  # small width
        splitter.setStretchFactor(1, 4)  # video bigger

        # initial 20/80 sizing
        self.setCentralWidget(splitter)
        self._splitter = splitter
        QtCore.QTimer.singleShot(0, self._apply_initial_split)

        self.on_roi_changed()  # initialize ROI
        self._timeline_dragging = False
        # Preload model before showing GUI
        self._load_model_from_config()
        self._populate_class_checkboxes()
        
        # Enable keyboard shortcuts
        self.setFocusPolicy(Qt.StrongFocus)

    def _apply_initial_split(self):
        total = self._splitter.size().width()
        left = int(total * 0.2)
        right = total - left
        self._splitter.setSizes([left, right])

    def on_select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if not path:
            return
        self.video_path = path
        
        # Stop existing threads
        self._stop_threads()
        
        # Close existing seek capture
        if self.seek_cap:
            self.seek_cap.release()
        
        # Setup video info
        self.seek_cap = cv2.VideoCapture(self.video_path)
        if not self.seek_cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to open video")
            return
        
        # Extract video properties for time-based navigation
        self.video_fps = self.seek_cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = 30.0  # Default fallback
        
        total_frames = int(self.seek_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = total_frames / self.video_fps
        
        # Timeline setup (time-based)
        self.timeline_slider.setMaximum(1000)  # 1000 steps for precise control
        self.timeline_slider.setValue(0)
        self.current_time = 0.0
        
        # Update time display
        self._update_time_display()
        print(f"Video loaded: {total_frames} frames, {self.video_duration:.2f}s duration, {self.video_fps:.2f} FPS")
        
        # Reset traffic status for new video
        self._reset_traffic_status()
        
        # Show first frame
        ret, frame = self.seek_cap.read()
        if ret:
            self.video_canvas.set_frame(frame)

    def on_start(self):
        if not self.video_path:
            QtWidgets.QMessageBox.information(self, "Info", "Select a video first")
            return

        try:
            # Stop existing thread
            self._stop_threads()

            print(f"Starting simplified video processing: {self.video_path}")

            # Process events to keep UI responsive
            QtWidgets.QApplication.processEvents()

            # Create simple single-threaded processor
            self.video_processor_thread = SimpleVideoProcessor(
                self.video_path, 
                self.detector, 
                self.video_canvas.roi,
                start_frame=0,
                frame_skip_interval=self.frame_skip_interval
            )

            # Connect signals
            self.video_processor_thread.frame_ready.connect(self._on_frame_with_detections)

            # Start thread
            self.video_processor_thread.start()

            # Mark as playing
            self.is_playing = True
            self._update_playback_status()

            print("Video processing started successfully")

        except Exception as e:
            print(f"Error starting video processing: {e}")
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to start video processing: {str(e)}")
            self._stop_threads()

    def _on_frame_with_detections(self, frame, tracked_detections, frame_number):
        """Handle frame with detections from simple processor"""
        try:
            # Update timeline
            if not self._timeline_dragging and self.is_playing:
                current_time = frame_number / self.video_fps
                self.current_time = current_time
                slider_value = int((current_time / self.video_duration) * 1000) if self.video_duration > 0 else 0
                slider_value = max(0, min(1000, slider_value))
                self.timeline_slider.blockSignals(True)
                self.timeline_slider.setValue(slider_value)
                self.timeline_slider.blockSignals(False)
                self._update_time_display()
            
            # Process truck monitoring - ALWAYS call this to update truck state
            # Even if no detections, we need to update states (e.g., when person leaves)
                self._process_truck_monitoring(tracked_detections, frame)
            
            # Draw detections
            if tracked_detections and self.detector:
                frame = self._draw_detections_with_ids(frame, tracked_detections)
                avg_conf = sum(d[4] for d in tracked_detections) / len(tracked_detections)
                self.summary_label.setText(f"Detections: {len(tracked_detections)} | Avg conf: {avg_conf:.2f}")
            else:
                self.summary_label.setText("Detections: 0 | Avg conf: -")
            
            # Update display
            self.video_canvas.set_frame(frame)
            
        except Exception as e:
            print(f"Error in _on_frame_with_detections: {e}")
    
    def on_stop(self):
        self.is_playing = False
        self._stop_threads()
        self._update_playback_status()

    def on_toggle_fullscreen(self):
        # Toggle fullscreen with traffic light and status overlay
        if not self.fullscreen:
            self._controls_visible = self._splitter.widget(0).isVisible()
            self._splitter.widget(0).setVisible(False)
            self.showFullScreen()
            self.fullscreen = True
            # Add fullscreen overlay with traffic light and status
            QTimer.singleShot(150, self._add_fullscreen_overlay)  # Delay to ensure fullscreen is ready
        else:
            self._remove_fullscreen_overlay()
            self._splitter.widget(0).setVisible(True)
            self.showNormal()
            self.fullscreen = False
        
    def _add_fullscreen_overlay(self):
        """Add traffic light and status overlay in fullscreen mode"""
        # Create a single top bar to host all overlay widgets
        self.fullscreen_bar = QWidget(self)
        self.fullscreen_bar.setStyleSheet("""
            QWidget { background-color: rgba(0, 0, 0, 200); }
        """)
        bar_layout = QHBoxLayout()
        bar_layout.setContentsMargins(40, 20, 40, 20)
        bar_layout.setSpacing(35)
        self.fullscreen_bar.setLayout(bar_layout)

        # Add left spacer
        bar_layout.addStretch(1)
        
        # Traffic light widget (left side of center) - LARGER
        self.fullscreen_traffic_light = QWidget(self.fullscreen_bar)
        self.fullscreen_traffic_light.setStyleSheet("""
            QWidget { background-color: rgba(0,0,0,250); border: 4px solid rgba(255,255,255,240); border-radius: 20px; }
        """)
        light_layout = QHBoxLayout()
        light_layout.setContentsMargins(25, 18, 25, 18)
        light_layout.setSpacing(25)
        self.fullscreen_red_light = QLabel("●", self.fullscreen_traffic_light)
        self.fullscreen_yellow_light = QLabel("●", self.fullscreen_traffic_light)
        self.fullscreen_green_light = QLabel("●", self.fullscreen_traffic_light)
        for light in [self.fullscreen_red_light, self.fullscreen_yellow_light, self.fullscreen_green_light]:
            light.setAlignment(Qt.AlignCenter)
            light.setFixedSize(55, 55)
            light.setStyleSheet(""" QLabel { font-size: 42px; border-radius: 27px; background-color: #1a1a1a; color: #666; } """)
            light_layout.addWidget(light)
        self.fullscreen_traffic_light.setLayout(light_layout)
        bar_layout.addWidget(self.fullscreen_traffic_light)
        
        # Small spacer
        bar_layout.addSpacing(35)
        
        # Status label (center-right) - LARGER
        self.fullscreen_status_label = QLabel(self.fullscreen_bar)
        self.fullscreen_status_label.setAlignment(Qt.AlignCenter)
        self.fullscreen_status_label.setStyleSheet("""
            QLabel { font-size: 22px; font-weight: bold; padding: 16px 28px; border: 4px solid rgba(255,255,255,240); border-radius: 12px; background-color: rgba(255,255,255,255); color: #000; }
        """)
        bar_layout.addWidget(self.fullscreen_status_label)
        
        # Add right spacer
        bar_layout.addStretch(1)
        
        # Exit button (far right) - LARGER
        self.exit_fullscreen_btn = QPushButton("✕", self.fullscreen_bar)
        self.exit_fullscreen_btn.setFixedSize(55, 55)
        self.exit_fullscreen_btn.setStyleSheet("""
            QPushButton { background-color: rgba(30,30,30,250); color: #fff; border: 4px solid #fff; border-radius: 27px; font-size: 28px; font-weight: bold; }
            QPushButton:hover { background-color: rgba(255,0,0,250); border-color: rgba(255,0,0,255); }
        """)
        self.exit_fullscreen_btn.clicked.connect(self.on_toggle_fullscreen)
        bar_layout.addWidget(self.exit_fullscreen_btn)

        # Show and position the bar
        self._update_fullscreen_overlay_position()
        self.fullscreen_bar.show()
        self.fullscreen_bar.raise_()  # Make sure bar is on top
        print(f"Fullscreen bar created and shown")

        # Initialize state
        self._sync_fullscreen_traffic_light()
        self._sync_fullscreen_status()
        
    def _remove_fullscreen_overlay(self):
        """Remove fullscreen overlay widgets"""
        if hasattr(self, 'fullscreen_bar'):
            self.fullscreen_bar.deleteLater()
            delattr(self, 'fullscreen_bar')
        for attr in [
            'fullscreen_traffic_light','fullscreen_red_light','fullscreen_yellow_light',
            'fullscreen_green_light','fullscreen_status_label','exit_fullscreen_btn']:
            if hasattr(self, attr):
                try:
                    getattr(self, attr).deleteLater()
                except Exception:
                    pass
                delattr(self, attr)
    
    def _update_fullscreen_overlay_position(self):
        """Update position of fullscreen overlay widgets"""
        if not self.fullscreen:
            return
        if hasattr(self, 'fullscreen_bar'):
            self.fullscreen_bar.setGeometry(0, 0, self.width(), 140)
            self.fullscreen_bar.raise_()  # Make sure it's on top
            print(f"Fullscreen bar positioned: 0, 0, {self.width()}, 140")
    
    def _sync_fullscreen_traffic_light(self):
        """Sync fullscreen traffic light with main traffic light"""
        if not hasattr(self, 'fullscreen_red_light'):
            return
        
        # Get current active light from main traffic light
        active_light = None
        if self.red_light.styleSheet().find("#ff4444") != -1:
            active_light = "red"
        elif self.yellow_light.styleSheet().find("#ffff44") != -1:
            active_light = "yellow"
        elif self.green_light.styleSheet().find("#44ff44") != -1:
            active_light = "green"
        
        # Reset all lights
        self.fullscreen_red_light.setStyleSheet("""
            QLabel {
                font-size: 25px;
                border-radius: 20px;
                background-color: #404040;
                color: #666666;
            }
        """)
        self.fullscreen_yellow_light.setStyleSheet("""
            QLabel {
                font-size: 25px;
                border-radius: 20px;
                background-color: #404040;
                color: #666666;
            }
        """)
        self.fullscreen_green_light.setStyleSheet("""
            QLabel {
                font-size: 25px;
                border-radius: 20px;
                background-color: #404040;
                color: #666666;
            }
        """)
        
        # Set active light with more visible styling (LARGER)
        if active_light == "red":
            self.fullscreen_red_light.setStyleSheet("""
                QLabel {
                    font-size: 42px;
                    border-radius: 27px;
                    background-color: #ff0000;
                    color: #ffffff;
                    border: 4px solid #cc0000;
                }
            """)
        elif active_light == "yellow":
            self.fullscreen_yellow_light.setStyleSheet("""
                QLabel {
                    font-size: 42px;
                    border-radius: 27px;
                    background-color: #ffff00;
                    color: #000000;
                    border: 4px solid #cccc00;
                }
            """)
        elif active_light == "green":
            self.fullscreen_green_light.setStyleSheet("""
                QLabel {
                    font-size: 42px;
                    border-radius: 27px;
                    background-color: #00ff00;
                    color: #000000;
                    border: 4px solid #00cc00;
                }
            """)
    
    def _sync_fullscreen_status(self):
        """Sync fullscreen status label with main status"""
        if hasattr(self, 'fullscreen_status_label'):
            self.fullscreen_status_label.setText(self.status_label.text())

        
    def on_roi_changed(self):
        try:
            # Get 5 corner coordinates
            corner1_x = int(self.input_corner1_x.text())
            corner1_y = int(self.input_corner1_y.text())
            corner2_x = int(self.input_corner2_x.text())
            corner2_y = int(self.input_corner2_y.text())
            corner3_x = int(self.input_corner3_x.text())
            corner3_y = int(self.input_corner3_y.text())
            corner4_x = int(self.input_corner4_x.text())
            corner4_y = int(self.input_corner4_y.text())
            corner5_x = int(self.input_corner5_x.text())
            corner5_y = int(self.input_corner5_y.text())
            
            # Get parking line coordinates
            parking_line1_x = int(self.input_parking_line1_x.text())
            parking_line1_y = int(self.input_parking_line1_y.text())
            parking_line2_x = int(self.input_parking_line2_x.text())
            parking_line2_y = int(self.input_parking_line2_y.text())
            
            # Create ROI as list of 5 corners
            roi_corners = [(corner1_x, corner1_y), (corner2_x, corner2_y), 
                          (corner3_x, corner3_y), (corner4_x, corner4_y),
                          (corner5_x, corner5_y)]
            
            # For compatibility with existing code, also calculate bounding box
            x_coords = [corner1_x, corner2_x, corner3_x, corner4_x, corner5_x]
            y_coords = [corner1_y, corner2_y, corner3_y, corner4_y, corner5_y]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            
            # Set both the polygon corners and bounding box
            self.video_canvas.set_roi_corners(roi_corners)
            self.video_canvas.set_roi((x1, y1, x2, y2))
            
            # Set parking line coordinates
            self.parking_line_points = [(parking_line1_x, parking_line1_y), (parking_line2_x, parking_line2_y)]
            self.video_canvas.set_parking_line_points(self.parking_line_points)
            # Calculate average Y position for truck crossing detection
            self.parking_line_y = (parking_line1_y + parking_line2_y) // 2
            
            # Note: We don't pass ROI to video processor anymore because
            # detection now runs on full frame to properly detect partial objects
        except ValueError:
            pass
            
    # Removed _on_frame_ready - now using _on_frame_with_detections
            
    def _on_detection_complete(self, frame, detections, frame_number):
        """Handle detection complete signal - send to tracking thread"""
        # Send detections to tracking thread
        if self.tracking_thread:
            self.tracking_thread.add_detection(frame, detections, frame_number)
            
    def _on_tracking_complete(self, frame, tracked_detections, frame_number):
        """Handle tracking complete signal - final processing"""
        # Process truck monitoring logic for traffic status
        self._process_truck_monitoring(tracked_detections, frame)
        
        # Update frame with detections
        frame_with_detections = self._draw_detections_with_ids(frame, tracked_detections)
        self.video_canvas.set_frame(frame_with_detections)
        
        # Update detection summary
        if tracked_detections:
            avg_conf = sum(d[4] for d in tracked_detections) / len(tracked_detections)
            self.summary_label.setText(f"Detections: {len(tracked_detections)} | Avg conf: {avg_conf:.2f}")
        else:
            self.summary_label.setText("Detections: 0 | Avg conf: -")
        
    # _on_detection_ready removed - functionality merged into _on_frame_with_detections
        
    def _stop_threads(self):
        try:
            self.is_playing = False
            
            # Stop simple video processor
            if self.video_processor_thread:
                try:
                    self.video_processor_thread.stop()
                    self.video_processor_thread.quit()
                    if not self.video_processor_thread.wait(500):
                        print("Video processor thread did not stop gracefully")
                except Exception as e:
                    print(f"Error stopping video processor: {e}")
                finally:
                    self.video_processor_thread = None
                    
            # Stop YOLO processor if exists
            if hasattr(self, 'yolo_processor') and self.yolo_processor:
                try:
                    self.yolo_processor.stop()
                    self.yolo_processor.quit()
                    if not self.yolo_processor.wait(500):
                        print("YOLO processor thread did not stop gracefully")
                except Exception as e:
                    print(f"Error stopping yolo processor: {e}")
                finally:
                    self.yolo_processor = None
                
            # Clear buffer safely
            try:
                while not self.buffer.empty():
                    try:
                        self.buffer.get_nowait()
                        self.buffer.task_done()
                    except (queue.Empty, AttributeError):
                        break
            except Exception as e:
                print(f"Error clearing buffer: {e}")
                    
            # Cancel any pending operations
            self.pending_seek_time = None
            try:
                if self.seek_timer.isActive():
                    self.seek_timer.stop()
                if self.scroll_debounce_timer.isActive():
                    self.scroll_debounce_timer.stop()
            except:
                pass
                
            self._update_playback_status()
        except Exception as e:
            print(f"Error stopping threads: {e}")
            import traceback
            traceback.print_exc()


    def _update_time_display(self):
        """Update the time display label"""
        current_min = int(self.current_time // 60)
        current_sec = int(self.current_time % 60)
        total_min = int(self.video_duration // 60)
        total_sec = int(self.video_duration % 60)
        
        self.time_info_label.setText(f"Time: {current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")

    def _update_playback_status(self):
        """Update the playback status indicator"""
        if self.is_playing:
            self.playback_status_label.setText("Status: Playing")
            self.playback_status_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    font-weight: bold;
                    padding: 5px;
                    border: 1px solid #4CAF50;
                    border-radius: 3px;
                    background-color: #E8F5E8;
                    color: #2E7D32;
                }
            """)
        else:
            self.playback_status_label.setText("Status: Stopped")
            self.playback_status_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    font-weight: bold;
                    padding: 5px;
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    background-color: #fff;
                }
            """)
    
    def _on_timeline_pressed(self):
        """Timeline pressed - pause video and set dragging state"""
        try:
            self.operation_mutex.lock()
            if self.operation_lock:
                self.operation_mutex.unlock()
                return
            self.operation_lock = True
            self.operation_mutex.unlock()
            
            self.is_playing = False
            self._stop_threads()
            self._timeline_dragging = True
            
            # Cancel any pending seeks and timers
            self.pending_seek_time = None
            if self.seek_timer.isActive():
                self.seek_timer.stop()
            if self.scroll_debounce_timer.isActive():
                self.scroll_debounce_timer.stop()
            
            print(f"Timeline pressed at time: {self.current_time:.2f}s")
            
            self.operation_mutex.lock()
            self.operation_lock = False
            self.operation_mutex.unlock()
        except Exception as e:
            print(f"Error in timeline pressed: {e}")
            try:
                self.operation_mutex.lock()
                self.operation_lock = False
                self.operation_mutex.unlock()
            except:
                pass

    def _on_timeline_released(self):
        """Timeline released - show frame with detection and clear dragging state"""
        try:
            self.operation_mutex.lock()
            if self.operation_lock:
                self.operation_mutex.unlock()
                return
            self.operation_lock = True
            self.operation_mutex.unlock()
            
            self._timeline_dragging = False
            print(f"Timeline released at time: {self.current_time:.2f}s")
            
            # Show frame with detection (this will trigger truck monitoring)
            self._show_frame_at_time(self.current_time)
            
            # Auto-start video from current position if not already playing
            if not self.is_playing and self.video_path:
                # Use a longer delay to prevent conflicts
                QTimer.singleShot(500, self._safe_start_video)
            else:
                # Force update traffic status even if not playing
                self._force_update_traffic_status()
            
            self.operation_mutex.lock()
            self.operation_lock = False
            self.operation_mutex.unlock()
        except Exception as e:
            print(f"Error in timeline released: {e}")
            try:
                self.operation_mutex.lock()
                self.operation_lock = False
                self.operation_mutex.unlock()
            except:
                pass

    def _on_timeline_changed(self, value):
        """Timeline changed - show frame immediately with debouncing"""
        try:
            if not self.video_path or not self.seek_cap or not self.seek_cap.isOpened() or self.video_duration <= 0:
                return
                
            self.operation_mutex.lock()
            if self.operation_lock:
                self.operation_mutex.unlock()
                return
            self.operation_mutex.unlock()
                
            # Convert slider value to time
            time_ratio = value / 1000.0
            self.current_time = time_ratio * self.video_duration
            
            # Update time display
            self._update_time_display()
            
            # Use debouncing to prevent excessive seeking during rapid scrolling
            self.scroll_debounce_timer.stop()
            self.scroll_debounce_timer.start(100)  # 100ms debounce
        except Exception as e:
            print(f"Error in timeline changed: {e}")
    
    def _process_debounced_scroll(self):
        """Process the debounced scroll operation"""
        try:
            if not self.video_path or not self.seek_cap or not self.seek_cap.isOpened():
                return
                
            # Show frame with detection when scrolling stops
            self._show_frame_at_time(self.current_time)
        except Exception as e:
            print(f"Error in debounced scroll: {e}")


    def _draw_detections_with_ids(self, frame, detections):
        """Draw detections with unique IDs and confidence scores
        
        Only draws bounding boxes that touch or are inside the ROI.
        If object is partially in ROI, draws the FULL bounding box.
        """
        if not detections or not self.detector:
            return frame
            
        # Create a copy of the frame to draw on
        frame_copy = frame.copy()
        
        # Get ROI bounds for filtering
        roi = self.video_canvas.roi
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
            class_name = self.model_names.get(class_id, f"Class_{class_id}")
            
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
            
            # Bottom-center dot removed for cleaner display
        
        return frame_copy
        
    def _assign_unique_ids(self, detections):
        """Assign unique IDs to detections based on position and class"""
        if not self.tracking_enabled:
            return detections
            
        current_frame_objects = []
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection[:6]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Find closest existing tracked object
            best_match_id = None
            best_distance = float('inf')
            
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data['class_id'] == class_id:
                    # Calculate distance from last known position
                    last_pos = obj_data['last_position']
                    distance = ((center_x - last_pos[0])**2 + (center_y - last_pos[1])**2)**0.5
                    
                    if distance < self.tracking_distance_threshold and distance < best_distance:
                        best_distance = distance
                        best_match_id = obj_id
            
            # Assign ID
            if best_match_id is not None:
                # Update existing tracked object
                obj_id = best_match_id
                self.tracked_objects[obj_id]['last_position'] = (center_x, center_y)
                self.tracked_objects[obj_id]['frames_since_last_seen'] = 0
                self.tracked_objects[obj_id]['last_detection'] = detection
            else:
                # Create new tracked object
                self.detection_id_counter += 1
                obj_id = self.detection_id_counter
                self.tracked_objects[obj_id] = {
                    'class_id': class_id,
                    'last_position': (center_x, center_y),
                    'frames_since_last_seen': 0,
                    'first_seen_frame': self.detection_id_counter,
                    'last_detection': detection
                }
            
            # Add ID to detection
            detection_with_id = detection + (obj_id,)
            current_frame_objects.append(detection_with_id)
        
        # Update frames_since_last_seen for all tracked objects
        for obj_id in list(self.tracked_objects.keys()):
            self.tracked_objects[obj_id]['frames_since_last_seen'] += 1
            
            # Remove old objects
            if self.tracked_objects[obj_id]['frames_since_last_seen'] > self.max_tracking_frames:
                del self.tracked_objects[obj_id]
        
        return current_frame_objects
        
    def _process_truck_monitoring(self, detections, frame):
        """Main truck monitoring logic"""
        # Store debug data for periodic printing
        self.last_debug_data = {
            'detections': len(detections) if detections else 0,
            'truck_state': self.truck_state,
            'roi': self.video_canvas.roi,
            'parking_line_y': self.parking_line_y,
            'detections_detail': detections if detections else []
        }
        
        # Format: (x1, y1, x2, y2, conf, class_id, track_id) or (x1, y1, x2, y2, conf, class_id)
        
        # Separate trucks, persons, and forklifts
        trucks = [d for d in detections if d[5] == 2]  # Class 2 is truck
        persons = [d for d in detections if d[5] == 0]  # Class 0 is person
        forklifts = [d for d in detections if d[5] == 1]  # Class 1 is forklift
        
        # STEP 1: Update Truck State
        self._update_truck_state(trucks)
        
        # STEP 2: Check Person Violations (returns boolean)
        has_violation = self._check_person_violations(persons, trucks)
        
        # STEP 3: Check for persons/forklifts in ROI when no truck
        has_person_or_forklift_in_roi = self._check_person_or_forklift_in_roi(persons, forklifts)
        
        # STEP 4: Update UI
        self._update_status_display(has_violation, has_person_or_forklift_in_roi)

    def _update_truck_state(self, trucks):
        """Update truck state based on detection and position"""
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
        roi = self.video_canvas.roi
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

    def _check_person_violations(self, persons, trucks):
        """Check for person violations - returns True if violation exists
        
        IMPORTANT: Only checks bottom-center dot (feet position) of person bounding box.
        Even if upper body is in ROI, if bottom-center dot is NOT in ROI, it's NOT a violation.
        This handles side-angle camera views where walkway is behind ROI boundary.
        """
        
        # Store person dot data for periodic printing
        person_dots = []
        for person in persons if persons else []:
            px1, py1, px2, py2 = person[:4]
            person_bottom_center_x = (px1 + px2) / 2
            person_bottom_y = py2
            person_dots.append((person_bottom_center_x, person_bottom_y))
        
        self.last_debug_data['person_dots'] = person_dots
        
        # NO VIOLATION if no truck detected or truck state is NO_TRUCK
        if not trucks or len(trucks) == 0 or self.truck_state == "NO_TRUCK":
            return False
        
        # NO VIOLATION if truck is PARKED
        if self.truck_state == "PARKED":
            return False
        
        # Check if ROI is set
        if not self.video_canvas.roi:
            return False
        
        # Get ROI as polygon (5 corners) for accurate checking
        roi_corners = self.video_canvas.roi_corners if hasattr(self.video_canvas, 'roi_corners') else None
        
        if roi_corners is None or len(roi_corners) < 3:
            # Fallback to bounding box
            roi_x1, roi_y1, roi_x2, roi_y2 = self.video_canvas.roi
            use_polygon = False
        else:
            use_polygon = True
        
        # NO VIOLATION if no persons detected
        if not persons or len(persons) == 0:
            return False
        
        # Check each person: ONLY bottom-center dot being in ROI counts as violation
        # Upper body in ROI but bottom-center dot outside ROI = NOT a violation (yellow light)
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
                roi_x1, roi_y1, roi_x2, roi_y2 = self.video_canvas.roi
                person_bottom_in_roi = (roi_x1 <= person_bottom_center_x <= roi_x2 and 
                            roi_y1 <= person_bottom_y <= roi_y2)
            
            # Optional: Check if any part of person bbox overlaps ROI (for debug info only)
            # Get bbox coordinates for overlap check
            if use_polygon:
                # For polygon check, we need to get bbox from self.video_canvas.roi
                bbox_roi_x1, bbox_roi_y1, bbox_roi_x2, bbox_roi_y2 = self.video_canvas.roi
                person_bbox_overlaps_roi = not (px2 < bbox_roi_x1 or px1 > bbox_roi_x2 or py2 < bbox_roi_y1 or py1 > bbox_roi_y2)
            else:
                person_bbox_overlaps_roi = not (px2 < roi_x1 or px1 > roi_x2 or py2 < roi_y1 or py1 > roi_y2)
            
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
    
    def _check_person_or_forklift_in_roi(self, persons, forklifts):
        """Check if any person or forklift is in ROI"""
        if not self.video_canvas.roi:
            return False
        
        roi_x1, roi_y1, roi_x2, roi_y2 = self.video_canvas.roi
        
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

    def _print_debug_info(self):
        """Print debug info every second"""
        if not self.last_debug_data:
            return
        
        roi = self.last_debug_data.get('roi', None)
        person_dots = self.last_debug_data.get('person_dots', [])
        truck_state = self.last_debug_data.get('truck_state', 'UNKNOWN')
        
        print("\n" + "="*60)
        print(f"DEBUG (Every 1 sec) - Truck State: {truck_state}")
        print(f"ROI: {roi}")
        print(f"Person Bottom-Center Dots: {person_dots}")
        
        if roi and person_dots:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi
            for i, (dot_x, dot_y) in enumerate(person_dots):
                in_roi = (roi_x1 <= dot_x <= roi_x2 and roi_y1 <= dot_y <= roi_y2)
                status = "IN ROI" if in_roi else "OUT OF ROI"
                print(f"  Person {i+1} Dot: ({dot_x:.1f}, {dot_y:.1f}) - {status}")
        
        print("="*60 + "\n")
    
    def _update_status_display(self, has_violation, has_person_or_forklift_in_roi):
        """Update status display and traffic light
        
        Light Logic:
        - Green: No truck detected OR truck is PARKED (safe)
        - Yellow: Truck detected (not PARKED) AND no violation (person bottom-center dot NOT in ROI)
        - Red: Truck detected AND violation (person bottom-center dot IS in ROI)
        
        Note: Even if person's upper body is in ROI, if bottom-center dot is NOT in ROI,
        it's NOT a violation (yellow light). This handles side-angle camera views.
        """
        
        # Store violation status in debug data
        self.last_debug_data['has_violation'] = has_violation
        
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
        
        # Update traffic light
        self._set_traffic_light(active_light)
        
        self.status_label.setText(status_text)
        
        # Sync fullscreen overlay if in fullscreen mode
        if self.fullscreen:
            self._sync_fullscreen_traffic_light()
            self._sync_fullscreen_status()
        

    def _set_traffic_light(self, active_light):
        """Set the active traffic light (red, yellow, green)"""
        # Reset all lights to grey
        self.red_light.setStyleSheet("""
            QLabel {
                font-size: 30px;
                border-radius: 25px;
                background-color: #404040;
                color: #666666;
            }
        """)
        self.yellow_light.setStyleSheet("""
            QLabel {
                font-size: 30px;
                border-radius: 25px;
                background-color: #404040;
                color: #666666;
            }
        """)
        self.green_light.setStyleSheet("""
            QLabel {
                font-size: 30px;
                border-radius: 25px;
                background-color: #404040;
                color: #666666;
            }
        """)
        
        # Set the active light
        if active_light == "red":
            self.red_light.setStyleSheet("""
                QLabel {
                    font-size: 30px;
                    border-radius: 25px;
                    background-color: #ff4444;
                    color: #ffffff;
                    border: 2px solid #cc0000;
                }
            """)
        elif active_light == "yellow":
            self.yellow_light.setStyleSheet("""
                QLabel {
                    font-size: 30px;
                    border-radius: 25px;
                    background-color: #ffff44;
                    color: #000000;
                    border: 2px solid #cccc00;
                }
            """)
        elif active_light == "green":
            self.green_light.setStyleSheet("""
                QLabel {
                    font-size: 30px;
                    border-radius: 25px;
                    background-color: #44ff44;
                    color: #000000;
                    border: 2px solid #00cc00;
                }
            """)

    def _reset_traffic_status(self):
        """Reset traffic status to initial state"""
        self.truck_state = "NO_TRUCK"
        self.last_truck_position = None
        self.parking_timer_start = None
        self._update_status_display(False, False)
        print("Traffic status reset to initial state")

    def _force_update_traffic_status(self):
        """Force update traffic status for current frame"""
        try:
            if not self.video_path or not self.seek_cap or not self.detector:
                print("Cannot force update: No video or detector available")
                return
                
            # Get current frame
            current_frame_idx = int(self.current_time * self.video_fps)
            self.seek_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = self.seek_cap.read()
            
            if ret:
                # Run detection on current frame
                detections = self.detector.detect(frame)
                if detections:
                    selected_ids = self._selected_class_ids()
                    if selected_ids is not None:
                        detections = [d for d in detections if d[5] in selected_ids]
                
                # Process truck monitoring logic
                self._process_truck_monitoring(detections, frame)
                
                print(f"Force updated traffic status for frame at time {self.current_time:.2f}s")
            else:
                print("Failed to read frame for force update")
        except Exception as e:
            print(f"Error force updating status: {e}")


    def _refresh_current_status(self):
        """Manually refresh traffic status for current frame"""
        if not self.video_path or not self.seek_cap or not self.detector:
            print("Cannot refresh status: No video or detector available")
            return
            
        try:
            # Get current frame
            current_frame_idx = int(self.current_time * self.video_fps)
            self.seek_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = self.seek_cap.read()
            
            if ret:
                # Run detection on current frame
                detections = self.detector.detect(frame)
                if detections:
                    selected_ids = self._selected_class_ids()
                    if selected_ids is not None:
                        detections = [d for d in detections if d[5] in selected_ids]
                
                # Process truck monitoring logic
                self._process_truck_monitoring(detections, frame)
                
                print(f"Traffic status refreshed for frame at time {self.current_time:.2f}s")
            else:
                print("Failed to read frame for status refresh")
        except Exception as e:
            print(f"Error refreshing status: {e}")

    def _process_seek_queue(self):
        """Process pending seek operation"""
        try:
            if self.pending_seek_time is not None and not self.seek_in_progress:
                seek_time = self.pending_seek_time
                self.pending_seek_time = None
                self.seek_in_progress = True
                self._do_seek(seek_time)
                self.seek_in_progress = False
        except Exception as e:
            print(f"Error in seek queue processing: {e}")
            self.seek_in_progress = False
            self.pending_seek_time = None

    def _do_seek(self, time_seconds):
        """Perform the actual video seeking with improved error handling"""
        try:
            # Ensure video capture is valid
            if not self.seek_cap or not self.seek_cap.isOpened():
                print("Video capture not available for seeking")
                return
                
            # Validate time bounds
            if time_seconds < 0 or time_seconds > self.video_duration:
                print(f"Invalid seek time: {time_seconds:.2f}s")
                return
                
            # Use time-based seeking to avoid codec conflicts
            success = self.seek_cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
            if not success:
                print(f"Failed to set video position to {time_seconds:.2f}s")
                return
                
            ret, frame = self.seek_cap.read()
            
            if ret and frame is not None:
                # Show frame immediately without detection to avoid UI freeze
                self.video_canvas.set_frame(frame)
                
                # Run detection in background if detector is available
                if self.detector:
                    # Use QTimer to run detection asynchronously with minimal delay
                    QTimer.singleShot(50, lambda: self._run_detection_async(frame))
            else:
                print(f"Failed to read frame at time {time_seconds:.2f}s")
        except Exception as e:
            print(f"Seek error: {e}")

    def _show_frame_at_time(self, time_seconds):
        """Show frame at specific time - non-blocking version"""
        # Store the seek request and process it asynchronously
        self.pending_seek_time = time_seconds
        
        # Start timer to process seek (non-blocking)
        if not self.seek_timer.isActive():
            self.seek_timer.start(50)  # Minimal delay for responsive scrolling
            
    def _run_detection_async(self, frame):
        """Run detection asynchronously to avoid blocking UI"""
        if not self.detector or frame is None:
            return  # Skip if no detector or invalid frame
            
        try:
            print("Running detection async...")
            
            # Use threading to prevent UI blocking
            import threading
            detection_result = [None]
            detection_error = [None]
            
            def run_detection():
                try:
                    if self.detector and frame is not None:
                        detection_result[0] = self.detector.detect(frame)
                except Exception as e:
                    detection_error[0] = e
            
            # Start detection in a separate thread with timeout
            detection_thread = threading.Thread(target=run_detection)
            detection_thread.daemon = True
            detection_thread.start()
            detection_thread.join(timeout=0.2)  # 200ms timeout
            
            if detection_thread.is_alive():
                # Detection timed out, use empty result
                print("Detection timeout, using empty result")
                detections = []
            elif detection_error[0]:
                print(f"Detection error: {detection_error[0]}")
                detections = []
            else:
                detections = detection_result[0] or []
            
            print(f"Detection result: {len(detections) if detections else 0} detections")
            
            if detections:
                selected_ids = self._selected_class_ids()
                if selected_ids is not None:
                    detections = [d for d in detections if d[5] in selected_ids]
                
                print(f"Filtered detections: {len(detections)} detections")
                
                # Process truck monitoring logic for traffic status - ALWAYS UPDATE
                self._process_truck_monitoring(detections, frame)
                
                # Update frame with detections
                frame_with_detections = self._draw_detections_with_ids(frame, detections)
                self.video_canvas.set_frame(frame_with_detections)
                
                # Update detection summary
                if detections:
                    avg_conf = sum(d[4] for d in detections) / len(detections)
                    self.summary_label.setText(f"Detections: {len(detections)} | Avg conf: {avg_conf:.2f}")
                else:
                    self.summary_label.setText("Detections: 0 | Avg conf: -")
            else:
                # No detections - still process truck monitoring to reset states - ALWAYS UPDATE
                print("No detections found, processing truck monitoring...")
                self._process_truck_monitoring([], frame)
                # Update detection summary
                self.summary_label.setText("Detections: 0 | Avg conf: -")
        except Exception as e:
            print(f"Async detection error: {e}")


    def _safe_start_video(self):
        """Safely start video from current position with error handling"""
        try:
            if not self.video_path or self.is_playing:
                return
                
            print(f"Safe starting video from time: {self.current_time:.2f}s")
            self._start_from_current_position()
        except Exception as e:
            print(f"Error in safe start video: {e}")

    def _start_from_current_position(self):
        """Start video playback from current timeline position"""
        if not self.video_path or self.is_playing:
            return
            
        try:
            # Stop existing threads
            self._stop_threads()
            
            # Convert current time to frame number
            start_frame = int(self.current_time * self.video_fps)
            print(f"Starting video from time: {self.current_time:.2f}s (frame: {start_frame})")
            
            # Create simple video processor starting from current position
            self.video_processor_thread = SimpleVideoProcessor(
                self.video_path,
                self.detector,
                self.video_canvas.roi,
                start_frame=start_frame,
                frame_skip_interval=self.frame_skip_interval
            )
            
            # Connect signals
            self.video_processor_thread.frame_ready.connect(self._on_frame_with_detections)
            
            # Start thread
            self.video_processor_thread.start()
            
            # Mark as playing
            self.is_playing = True
            self._update_playback_status()
            
            print("Video started")
            
        except Exception as e:
            print(f"Error starting video from position: {e}")
            self._stop_threads()
            self.is_playing = False

    def _load_model_from_config(self):
        try:
            with open('config.json', 'r') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {
                "weights_path": "yolov5s.pt", 
                "inference_size": 640, 
                "confidence_threshold": 0.35, 
                "iou_threshold": 0.45,
                "detection_confidence_min": 0.3,
                "parking_duration_seconds": 5.0,
                "buffer_size": 20,
                "use_ocsort_tracking": True,
                "ocsort_max_age": 30,
                "ocsort_min_hits": 2,
                "ocsort_iou_threshold": 0.2,
                "ocsort_distance_threshold": 100,
                "ocsort_velocity_threshold": 50
            }
        
        # Load truck monitoring configuration
        self.detection_confidence_min = cfg.get("detection_confidence_min", 0.3)
        self.parking_duration_seconds = cfg.get("parking_duration_seconds", 5.0)
        
        # Load frame skip configuration
        self.frame_skip_interval = cfg.get("frame_skip_interval", 1)
        self.frame_skip_count = 0
        
        # Load OC-SORT configuration
        self.use_ocsort_tracking = cfg.get("use_ocsort_tracking", True)
        self.ocsort_max_age = cfg.get("ocsort_max_age", 30)
        self.ocsort_min_hits = cfg.get("ocsort_min_hits", 3)
        self.ocsort_iou_threshold = cfg.get("ocsort_iou_threshold", 0.3)
        self.ocsort_distance_threshold = cfg.get("ocsort_distance_threshold", 50)
        self.ocsort_velocity_threshold = cfg.get("ocsort_velocity_threshold", 10)
        
        self.model_label.setText("Loading model...")
        QtWidgets.QApplication.processEvents()
        
        try:
            self.detector = YoloV5Detector(
                weights_path=cfg.get("weights_path", "yolov5s.pt"),
                inference_size=int(cfg.get("inference_size", 640)),
                conf_thres=float(cfg.get("confidence_threshold", 0.35)),
                iou_thres=float(cfg.get("iou_threshold", 0.45)),
            )
            try:
                self.model_names = self.detector.get_class_names()
            except Exception:
                self.model_names = CLASS_NAMES
            self.model_label.setText("Model: YOLOv5 loaded (custom)")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.detector = None
            self.model_names = CLASS_NAMES
            self.model_label.setText("Model: Failed to load - Video only mode")

    def _populate_class_checkboxes(self):
        # clear existing
        for cb in getattr(self, 'class_checkboxes', []):
            cb.setParent(None)
        self.class_checkboxes = []
        # populate from model names
        for cls_id in sorted(self.model_names.keys()):
            name = self.model_names[cls_id]
            cb = QtWidgets.QCheckBox(f"{cls_id}: {name}")
            cb.setChecked(True)
            cb.setProperty('cls_id', int(cls_id))
            self.class_layout.addWidget(cb)
            self.class_checkboxes.append(cb)
        self.class_layout.addStretch(1)

    def _selected_class_ids(self):
        if not self.class_checkboxes:
            return None
        ids = []
        for cb in self.class_checkboxes:
            if cb.isChecked():
                ids.append(int(cb.property('cls_id')))
        return set(ids)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        
        if key == Qt.Key_Home:
            # Home key goes to beginning
            if self.video_duration > 0:
                self.current_time = 0.0
                self.timeline_slider.setValue(0)
                self._show_frame_at_time(0.0)
                self._update_time_display()
            event.accept()
        elif key == Qt.Key_End:
            # End key goes to end
            if self.video_duration > 0:
                self.current_time = self.video_duration
                self.timeline_slider.setValue(1000)
                self._show_frame_at_time(self.video_duration)
                self._update_time_display()
            event.accept()
        else:
            super().keyPressEvent(event)
        
    def resizeEvent(self, event):
        """Handle window resize event"""
        super().resizeEvent(event)
        # Update fullscreen overlay position when window is resized
        if hasattr(self, 'fullscreen') and self.fullscreen:
            if hasattr(self, 'fullscreen_bar'):
                self._update_fullscreen_overlay_position()
                self.fullscreen_bar.show()
                self.fullscreen_bar.raise_()
        
    def closeEvent(self, event):
        # Cleanup threads on close
        try:
            self._stop_threads()
        except Exception as e:
            print(f"Error stopping threads on close: {e}")
            
        # Stop all timers
        try:
            if self.seek_timer.isActive():
                self.seek_timer.stop()
            if self.scroll_debounce_timer.isActive():
                self.scroll_debounce_timer.stop()
            if self.debug_timer.isActive():
                self.debug_timer.stop()
        except Exception as e:
            print(f"Error stopping timers on close: {e}")
            
        # Close video capture
        try:
            if self.seek_cap:
                self.seek_cap.release()
                self.seek_cap = None
        except Exception as e:
            print(f"Error closing video capture on close: {e}")
            
        event.accept()



def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


