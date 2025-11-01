# MainWindow class extracted from main.py
# Imports
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QWidget, QFileDialog, QLabel, QPushButton, QLineEdit, QHBoxLayout, QVBoxLayout, QSplitter, QGroupBox, QFormLayout, QSizePolicy, QSlider)
from PyQt5.QtCore import Qt, QTimer, QMutex
import cv2
import time
import json
import threading
import os
from detector import YoloV5Detector, CLASS_NAMES
from canvas import VideoCanvas
from threads import SimpleVideoProcessor
from detection_utils import draw_detections_with_ids
from monitoring import TruckMonitor
from plc_controller import PLCThread

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
        self._was_playing_before_drag = False  # Track if video was playing before slider drag
        
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
        
        # Truck monitoring - use TruckMonitor class
        self.parking_line_y = None  # Will be set based on ROI
        self.parking_line_points = None  # Will be set based on user input
        self.parking_duration_seconds = None  # Will be loaded from config.json
        self.truck_monitor = None  # Will be initialized after config loads
        
        # PLC controller for physical traffic light
        self.plc_thread = None
        
        # Configuration

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
        
        # Load video from config and show first frame
        self._load_video_from_config()
        
        # Initialize and start PLC thread for physical traffic light
        try:
            self.plc_thread = PLCThread()
            self.plc_thread.start()
            print("PLC thread started")
        except Exception as e:
            print(f"Failed to start PLC thread: {e}")
            self.plc_thread = None
        
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
        """Handle frame with detections from simple processor - optimized for UI responsiveness"""
        try:
            # Quick timeline update (lightweight operations only)
            if not self._timeline_dragging and self.is_playing:
                current_time = frame_number / self.video_fps
                self.current_time = current_time
                slider_value = int((current_time / self.video_duration) * 1000) if self.video_duration > 0 else 0
                slider_value = max(0, min(1000, slider_value))
                
                # Only update slider if value changed significantly (reduce UI updates)
                if abs(self.timeline_slider.value() - slider_value) > 1:
                    self.timeline_slider.blockSignals(True)
                    self.timeline_slider.setValue(slider_value)
                    self.timeline_slider.blockSignals(False)
                    
                # Debounce time display update (update less frequently)
                if not hasattr(self, '_last_time_update') or time.time() - self._last_time_update > 0.1:
                    self._update_time_display()
                    self._last_time_update = time.time()
            
            # Draw detections first (lightweight) - show frame immediately for smooth playback
            if tracked_detections and self.detector:
                frame = self._draw_detections_with_ids(frame, tracked_detections)
            
            # Update display immediately (before heavy processing)
            self.video_canvas.set_frame(frame)
            
            # Process truck monitoring - use QTimer to defer heavy processing
            # This prevents blocking the signal handler
            if not hasattr(self, '_monitoring_timer'):
                self._monitoring_timer = QTimer()
                self._monitoring_timer.setSingleShot(True)
                self._monitoring_data = None
                
                def process_monitoring():
                    """Process monitoring logic in a deferred callback"""
                    try:
                        data = self._monitoring_data
                        if data:
                            detections, frame = data
                            self._process_truck_monitoring(detections, frame)
                            
                            # Update summary label (deferred)
                            if detections:
                                avg_conf = sum(d[4] for d in detections) / len(detections)
                                self.summary_label.setText(f"Detections: {len(detections)} | Avg conf: {avg_conf:.2f}")
                            else:
                                self.summary_label.setText("Detections: 0 | Avg conf: -")
                    except Exception as e:
                        print(f"Error in deferred monitoring: {e}")
                
                # Connect once
                self._monitoring_timer.timeout.connect(process_monitoring)
            
            # Store data and schedule processing
            self._monitoring_data = (tracked_detections, frame)
            if not self._monitoring_timer.isActive():
                self._monitoring_timer.start(10)  # Process after 10ms (non-blocking)
            
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
    # Removed _on_detection_ready - functionality merged into _on_frame_with_detections
        
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
            
            # Remember if video was playing before drag
            self._was_playing_before_drag = self.is_playing
            
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
            
            # Resume video playback if it was playing before drag, or auto-start if video path exists
            if self._was_playing_before_drag or (not self.is_playing and self.video_path):
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
        """Draw detections with unique IDs and confidence scores - uses utility function"""
        if not detections or not self.detector:
            return frame
        return draw_detections_with_ids(frame, detections, self.model_names, self.video_canvas.roi)
        
    def _process_truck_monitoring(self, detections, frame):
        """Main truck monitoring logic - uses TruckMonitor class"""
        if not self.truck_monitor:
            return
            
        # Format: (x1, y1, x2, y2, conf, class_id, track_id) or (x1, y1, x2, y2, conf, class_id)
        
        # Separate trucks, persons, and forklifts
        trucks = [d for d in detections if d[5] == 2]  # Class 2 is truck
        persons = [d for d in detections if d[5] == 0]  # Class 0 is person
        forklifts = [d for d in detections if d[5] == 1]  # Class 1 is forklift
        
        # Calculate person dots for debug info
        person_dots = []
        for person in persons if persons else []:
            px1, py1, px2, py2 = person[:4]
            person_bottom_center_x = (px1 + px2) / 2
            person_bottom_y = py2
            person_dots.append((person_bottom_center_x, person_bottom_y))
        
        # Store debug data for periodic printing
        self.last_debug_data = {
            'detections': len(detections) if detections else 0,
            'truck_state': self.truck_monitor.truck_state if self.truck_monitor else 'UNKNOWN',
            'roi': self.video_canvas.roi,
            'parking_line_y': self.parking_line_y,
            'detections_detail': detections if detections else [],
            'person_dots': person_dots
        }
        
        # Get ROI and ROI corners for monitoring
        roi = self.video_canvas.roi
        roi_corners = self.video_canvas.roi_corners if hasattr(self.video_canvas, 'roi_corners') else None
        
        # STEP 1: Update Truck State using TruckMonitor
        self.truck_monitor.update_truck_state(trucks, roi, self.parking_line_y)
        
        # STEP 2: Check Person Violations using TruckMonitor
        has_violation = self.truck_monitor.check_person_violations(persons, trucks, roi, roi_corners)
        
        # STEP 3: Check for persons/forklifts in ROI when no truck
        has_person_or_forklift_in_roi = self.truck_monitor.check_person_or_forklift_in_roi(persons, forklifts, roi)
        
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
        truck_state = self.truck_monitor.truck_state if self.truck_monitor else self.last_debug_data.get('truck_state', 'UNKNOWN')
        
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
        
        # Get status info from TruckMonitor
        if self.truck_monitor:
            active_light, status_text = self.truck_monitor.get_status_info(has_violation, has_person_or_forklift_in_roi)
        else:
            # Fallback if truck_monitor not initialized
            active_light = "green"
            status_text = "Status: Initializing..."
        
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
        
        # Update physical PLC light if PLC thread is running (truly non-blocking via queue)
        if self.plc_thread and self.plc_thread.is_alive():
            try:
                # This just puts state in queue - non-blocking operation
                self.plc_thread.update_light_state(active_light)
            except Exception as e:
                print(f"Error updating PLC light: {e}")

    def _reset_traffic_status(self):
        """Reset traffic status to initial state"""
        if self.truck_monitor:
            self.truck_monitor.reset()
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
            # Use threading to prevent UI blocking - NO JOIN to avoid blocking UI thread
            import threading
            detection_result = [None]
            detection_error = [None]
            
            def run_detection():
                try:
                    if self.detector and frame is not None:
                        detection_result[0] = self.detector.detect(frame)
                except Exception as e:
                    detection_error[0] = e
            
            def on_detection_complete():
                """Callback when detection completes - runs on main thread"""
                try:
                    if detection_error[0]:
                        print(f"Detection error: {detection_error[0]}")
                        detections = []
                    else:
                        detections = detection_result[0] or []
                    
                    if detections:
                        selected_ids = self._selected_class_ids()
                        if selected_ids is not None:
                            detections = [d for d in detections if d[5] in selected_ids]
                        
                        # Process truck monitoring logic for traffic status
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
                        # No detections - still process truck monitoring to reset states
                        self._process_truck_monitoring([], frame)
                        self.summary_label.setText("Detections: 0 | Avg conf: -")
                except Exception as e:
                    print(f"Error in detection callback: {e}")
            
            # Start detection in a separate thread (non-blocking)
            detection_thread = threading.Thread(target=run_detection)
            detection_thread.daemon = True
            detection_thread.start()
            
            # Use QTimer to check completion instead of blocking join()
            # Check if thread is done every 50ms (non-blocking)
            def check_thread():
                if not detection_thread.is_alive():
                    # Thread finished, process results
                    on_detection_complete()
                else:
                    # Still running, check again in 50ms
                    QTimer.singleShot(50, check_thread)
            
            # Start checking after 50ms
            QTimer.singleShot(50, check_thread)
            
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
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
            with open(config_path, 'r') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {
                "weights_path": "yolov5s.pt", 
                "inference_size": 640, 
                "confidence_threshold": 0.35, 
                "iou_threshold": 0.45,
                "parking_duration_seconds": 5.0,
                "frame_skip_interval": 1
            }
        
        # Load truck monitoring configuration
        self.parking_duration_seconds = cfg.get("parking_duration_seconds", 5.0)
        
        # Initialize truck monitor
        self.truck_monitor = TruckMonitor(parking_duration_seconds=self.parking_duration_seconds)
        
        # Load frame skip configuration
        self.frame_skip_interval = cfg.get("frame_skip_interval", 1)
        self.frame_skip_count = 0
        
        
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
    
    def _load_video_from_config(self):
        """Load video path from config file and show first frame if available"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
            with open(config_path, 'r') as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            return
        
        video_path = cfg.get("video_path")
        if video_path:
            # Make path absolute relative to project root (where config.json is)
            if not os.path.isabs(video_path):
                project_root = os.path.dirname(os.path.dirname(__file__))
                video_path = os.path.join(project_root, video_path)
            
            if os.path.exists(video_path):
                self.video_path = video_path
                
                # Setup video info
                self.seek_cap = cv2.VideoCapture(self.video_path)
                if not self.seek_cap.isOpened():
                    print(f"Failed to open video: {self.video_path}")
                    self.video_path = None
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
                print(f"Video loaded from config: {total_frames} frames, {self.video_duration:.2f}s duration, {self.video_fps:.2f} FPS")
                
                # Reset traffic status for new video
                self._reset_traffic_status()
                
                # Show first frame immediately
                ret, frame = self.seek_cap.read()
                if ret:
                    self.video_canvas.set_frame(frame)
                    print("First frame displayed")
                else:
                    print("Failed to read first frame")
            else:
                print(f"Video path in config does not exist: {video_path}")
        else:
            print("No video_path specified in config.json")

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
        
        # Stop PLC thread
        try:
            if self.plc_thread and self.plc_thread.is_alive():
                self.plc_thread.stop()
                self.plc_thread.join(timeout=2)
                print("PLC thread stopped")
        except Exception as e:
            print(f"Error stopping PLC thread on close: {e}")
            
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