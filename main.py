import cv2
import threading
import time

import yaml

from core_ai.detector import PersonDetector
from core_ai.reid_matcher import ReIDMatcher
from core_ai.tracker import PersonTracker
from database.db_handler import DBHandler
from services.identity_policy import is_same_position
from services.orchestrator import GlobalPremiseTracker
from services.rendering import draw_highlighted_text
from utils.camera_stream import IPVideoStream
from utils.image_logger import ImageLogger


DEFAULT_PROCESSING_CONFIG = {
    "ai_frame_width": 640,
    "ai_frame_height": 480,
    "detect_every_n_frames": 1,
    "reid_interval_frames": 10,
    "tracker_max_age": 30,
    "camera_buffer_width": None,
    "camera_buffer_height": None,
    "display": True,
}


class CameraWorker(threading.Thread):
    def __init__(self, cam_name, source, detector, reid_matcher, global_tracker, processing_config=None):
        super().__init__()
        self.cam_name, self.source = cam_name, source
        self.detector, self.reid_matcher, self.global_tracker = detector, reid_matcher, global_tracker
        self.logger = ImageLogger()
        self.identity_anchor, self.logged_ids, self.last_results = {}, set(), []
        self.processing_config = {**DEFAULT_PROCESSING_CONFIG, **(processing_config or {})}
        self.tracker = PersonTracker(max_age=int(self.processing_config["tracker_max_age"]))
        self.spatial_memory = {}
        self.last_tracked_objs = []

        self.running, self.frame_count = True, 0
        self.fps_start_time, self.fps_frame_count, self.displayed_fps = 0, 0, 0

    def _normalize_source(self):
        return int(self.source) if str(self.source).isdigit() else self.source

    def _run_detection_and_tracking(self, frame):
        ai_width = int(self.processing_config["ai_frame_width"])
        ai_height = int(self.processing_config["ai_frame_height"])
        detect_every_n_frames = max(1, int(self.processing_config["detect_every_n_frames"]))

        if self.frame_count % detect_every_n_frames == 0 or not self.last_tracked_objs:
            ai_frame = cv2.resize(frame, (ai_width, ai_height))
            detections = self.detector.detect(ai_frame)
            tracked_objs = self.tracker.update(detections, ai_frame)

            scale_x = frame.shape[1] / ai_width
            scale_y = frame.shape[0] / ai_height
            self.last_tracked_objs = [
                [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, conf, track_id]
                for x1, y1, x2, y2, conf, track_id in tracked_objs
            ]

        return self.last_tracked_objs

    def run(self):
        print(f"[INFO] Booting AI vision for {self.cam_name}...")
        self.stream = IPVideoStream(
            self._normalize_source(),
            width=self.processing_config["camera_buffer_width"],
            height=self.processing_config["camera_buffer_height"],
        ).start()
        self.fps_start_time = time.time()

        while self.running:
            now = time.time()
            self.fps_frame_count += 1
            if now - self.fps_start_time >= 1.0:
                self.displayed_fps, self.fps_frame_count, self.fps_start_time = self.fps_frame_count, 0, now

            ret, frame = self.stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self.frame_count += 1
            tracked_objs = self._run_detection_and_tracking(frame)
            current_tids = [int(obj[5]) for obj in tracked_objs]

            for anchor_key, p_id in list(self.identity_anchor.items()):
                if p_id in self.reid_matcher.merged_ids:
                    new_id = self.reid_matcher.merged_ids[p_id]
                    self.identity_anchor[anchor_key] = new_id
                    p_id = new_id

                track_id_str = anchor_key.split('_')[-1]
                if int(float(track_id_str)) not in current_tids:
                    self.reid_matcher.log_disappearance(p_id)
                    del self.identity_anchor[anchor_key]

            self.last_results = []
            reid_interval_frames = max(1, int(self.processing_config["reid_interval_frames"]))

            for obj in tracked_objs:
                x1, y1, x2, y2, conf, t_id = obj
                t_id = int(t_id)
                bbox = [x1, y1, x2, y2]
                anchor_key = f"{self.cam_name}_{t_id}"
                p_id = self.identity_anchor.get(anchor_key)

                if p_id is None:
                    for old_p_id, (old_bbox, old_time) in list(self.spatial_memory.items()):
                        if now - old_time < 2.0:
                            if is_same_position(old_bbox, bbox):
                                p_id = old_p_id
                                self.identity_anchor[anchor_key] = p_id
                                break
                        else:
                            del self.spatial_memory[old_p_id]

                if p_id is None or self.frame_count % reid_interval_frames == 0:
                    new_p_id = self.reid_matcher.identify_and_register(frame, bbox, t_id, self.cam_name, existing_id=p_id)
                    if "Scanning" not in str(new_p_id):
                        p_id = new_p_id
                        self.identity_anchor[anchor_key] = p_id

                if p_id is None:
                    p_id = "Scanning..."

                if "Scanning" not in str(p_id):
                    self.global_tracker.mark_seen(p_id, self.cam_name)
                    self.spatial_memory[p_id] = (bbox, now)

                if "Stranger" in str(p_id) and t_id not in self.logged_ids:
                    self.logger.log_stranger(frame, bbox, self.cam_name)
                    self.logged_ids.add(t_id)

                self.last_results.append((bbox, p_id, conf))

            if self.processing_config["display"]:
                for bbox, p_id, conf in self.last_results:
                    box_color = (0, 255, 0) if "Person" in str(p_id) else (0, 165, 255) if "Stranger" in str(p_id) else (0, 255, 255)
                    bg_color = (0, 100, 0) if "Person" in str(p_id) else (0, 80, 200) if "Stranger" in str(p_id) else (0, 150, 150)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, 2)
                    draw_highlighted_text(frame, f"{p_id} ({conf:.2f})", (int(bbox[0]), int(bbox[1]) - 10), (255, 255, 255), bg_color)

                draw_highlighted_text(frame, f"FPS: {self.displayed_fps}", (15, 30), (255, 255, 255), (0, 0, 0))
                cv2.imshow(f"Local AI Feed: {self.cam_name}", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

        self.stream.stop()


def build_camera_workers(config, detector, reid_matcher, global_tracker):
    workers = []
    global_processing = config.get("processing", {})

    for name, camera_config in config["cameras"].items():
        if isinstance(camera_config, dict):
            source = camera_config.get("source")
            processing_config = {**global_processing, **camera_config.get("processing", {})}
        else:
            source = camera_config
            processing_config = global_processing

        worker = CameraWorker(name, source, detector, reid_matcher, global_tracker, processing_config=processing_config)
        workers.append(worker)
    return workers


def main():
    print("[SYSTEM] Starting Local AI Surveillance...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    db = DBHandler(config["database"]["db_path"])
    detector = PersonDetector(config["models"]["detection_weights"], config=config.get("detector", {}))
    reid_matcher = ReIDMatcher(db, config=config.get("reid", {}))

    global_tracker = GlobalPremiseTracker(db, reid_matcher)
    global_tracker.start()

    workers = build_camera_workers(config, detector, reid_matcher, global_tracker)
    for worker in workers:
        worker.start()

    print("\n[ACTIVE] System running locally. Press 'q' on the video window to safely exit.\n")

    try:
        while any(w.is_alive() for w in workers):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Keyboard Interrupt received.")
    finally:
        print("\n[SYSTEM] Shutting down. Saving final data...")
        for w in workers:
            w.running = False
        global_tracker.running = False
        cv2.destroyAllWindows()
        db.close()
        print("[SYSTEM] Shutdown complete.")


if __name__ == "__main__":
    main()
