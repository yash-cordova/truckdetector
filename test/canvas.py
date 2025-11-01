"""Video canvas widget for displaying video frames with ROI and parking line overlays"""
import cv2
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt


class VideoCanvas(QLabel):
    """Widget for displaying video frames with ROI and parking line overlays"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._frame = None
        self.roi = None  # (x1, y1, x2, y2) - bounding box for compatibility
        self.roi_corners = None  # List of 5 corners for polygon
        self.parking_line_points = None

    def set_frame(self, frame_bgr):
        """Set the current frame to display"""
        self._frame = frame_bgr
        self.update()

    def set_roi(self, roi):
        """Set ROI bounding box"""
        self.roi = roi
        self.update()

    def set_roi_corners(self, corners):
        """Set ROI as polygon (5 corners)"""
        self.roi_corners = corners
        self.update()
        
    def set_parking_line_points(self, points):
        """Set parking line coordinates (2 points)"""
        self.parking_line_points = points
        self.update()

    def paintEvent(self, event):
        """Paint the frame with ROI and parking line overlays"""
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
            # Scale to fit
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

