import threading

import torch
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, weights_path, config=None):
        config = config or {}
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence = float(config.get("confidence", 0.35))
        self.iou = float(config.get("iou", 0.45))
        self.imgsz = int(config.get("imgsz", 640))
        self.max_det = int(config.get("max_det", 32))
        self.half = bool(config.get("half", self.device == "cuda"))
        self.model_lock = threading.Lock()

        self.model = YOLO(weights_path)
        print(f"[AI] YOLO Detector running on: {self.device} | imgsz={self.imgsz} | conf={self.confidence}")

    def detect(self, frame):
        with self.model_lock:
            results = self.model.predict(
                frame,
                classes=[0],
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.imgsz,
                max_det=self.max_det,
                half=self.half,
                device=self.device,
                verbose=False,
            )

        detections = []
        if not results or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
        return detections
