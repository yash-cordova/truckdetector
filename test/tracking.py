"""Tracking components: KalmanFilter and Track classes for object tracking"""
import numpy as np


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

