import threading
import time

import cv2


class IPVideoStream:
    def __init__(self, src=0, width=None, height=None, warmup_time=0.2):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        self.lock = threading.Lock()
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.warmup_time = warmup_time

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        if self.warmup_time:
            time.sleep(self.warmup_time)
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame
            if not grabbed:
                time.sleep(0.05)

    def read(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return self.grabbed, frame

    def stop(self):
        self.stopped = True
        self.stream.release()
