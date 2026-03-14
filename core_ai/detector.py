from ultralytics import YOLO
import torch

class PersonDetector:
    def __init__(self, weights_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(weights_path)
        print(f"[AI] YOLO Detector running on: {self.device}")

    def detect(self, frame):
        results = self.model.track(frame, classes=[0], persist=True, verbose=False)
        tracked_objs = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, track_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                tracked_objs.append([x1, y1, x2, y2, conf, track_id])
        return tracked_objs