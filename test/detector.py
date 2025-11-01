import threading
from typing import List, Tuple
import torch
import cv2


CLASS_NAMES = {0: "person", 1: "Forklift", 2: "truck"}


class YoloV5Detector:
    def __init__(self, weights_path: str = "yolov5s.pt", device: str = None, inference_size: int = 640, conf_thres: float = 0.35, iou_thres: float = 0.45):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_size = inference_size
        # Using torch hub to load yolov5
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
        self.model.to(self.device)
        self.model.conf = conf_thres
        self.model.iou = iou_thres

    def detect(self, frame_bgr, roi: Tuple[int, int, int, int] = None):
        img = frame_bgr
        if roi is not None:
            x1, y1, x2, y2 = roi
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            img = img[y1:y2, x1:x2]

        results = self.model(img, size=self.inference_size)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, xyxy)
            if roi is not None:
                # offset back
                ox1, oy1, _, _ = roi
                x1 += ox1
                x2 += ox1
                y1 += oy1
                y2 += oy1
            detections.append((x1, y1, x2, y2, float(conf), int(cls)))
        return detections

    @staticmethod
    def draw_detections(frame_bgr, detections: List[Tuple[int, int, int, int, float, int]], class_names=None):
        names = class_names if class_names is not None else CLASS_NAMES
        for x1, y1, x2, y2, conf, cls in detections:
            name = names.get(cls, str(cls))
            label = f"{name} {conf:.2f}"
            color = (0, 255, 255)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return frame_bgr

    def get_class_names(self):
        # Return a dict[int,str] mapping from the underlying model if available
        try:
            names = self.model.names  # usually list or dict
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            return {i: str(n) for i, n in enumerate(list(names))}
        except Exception:
            return CLASS_NAMES
