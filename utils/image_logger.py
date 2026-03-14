import cv2
import os
import time

class ImageLogger:
    def __init__(self, log_dir="logs/strangers"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_stranger(self, frame, bbox, cam_name):
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop is not None and crop.size > 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{self.log_dir}/{cam_name}_stranger_{timestamp}.jpg"
            cv2.imwrite(filename, crop)