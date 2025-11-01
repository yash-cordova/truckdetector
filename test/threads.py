"""Thread class for video processing"""
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex


class SimpleVideoProcessor(QThread):
    """Simple single-threaded video processor - no complex queues"""
    frame_ready = pyqtSignal(np.ndarray, list, int)  # frame, tracked_detections, frame_number
    
    def __init__(self, video_path, detector, start_frame=0, frame_skip_interval=1):
        super().__init__()
        self.video_path = video_path
        self.detector = detector
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

