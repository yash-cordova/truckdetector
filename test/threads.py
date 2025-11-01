"""Thread classes for video processing, detection, and tracking"""
import cv2
import queue
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition


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

