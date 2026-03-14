import cv2
import threading
import time
import yaml
import numpy as np
from database.db_handler import DBHandler
from core_ai.detector import PersonDetector
from core_ai.reid_matcher import ReIDMatcher
from utils.camera_stream import IPVideoStream
from utils.image_logger import ImageLogger

class GlobalPremiseTracker(threading.Thread):
    def __init__(self, db_handler, reid_matcher):
        super().__init__()
        self.db = db_handler
        self.reid = reid_matcher 
        self.active_visitors = {} 
        self.premise_timeout = 15.0 
        self.running = True
        self.lock = threading.Lock()

    def mark_seen(self, person_id, cam_name):
        now = time.time()
        with self.lock:
            if person_id not in self.active_visitors:
                self.active_visitors[person_id] = {"first_seen": now, "last_seen": now, "path": [cam_name]}
            else:
                self.active_visitors[person_id]["last_seen"] = now
                if self.active_visitors[person_id]["path"][-1] != cam_name:
                    self.active_visitors[person_id]["path"].append(cam_name)

    def run(self):
        merge_timer = 0
        while self.running:
            now = time.time()
            with self.lock:
                for p_id in list(self.active_visitors.keys()):
                    data = self.active_visitors[p_id]
                    if now - data["last_seen"] > self.premise_timeout:
                        duration = data["last_seen"] - data["first_seen"]
                        if duration >= 2.0:
                            path_str = " -> ".join(data["path"])
                            self.db.save_premise_log(p_id, path_str, data["first_seen"], data["last_seen"], duration)
                            print(f"[GLOBAL LOG] {p_id} left. Total time: {int(duration)}s. Path: {path_str}")
                        del self.active_visitors[p_id]
            
            merge_timer += 2
            if merge_timer >= 10:
                self.reid.run_merger_pass()
                merge_timer = 0
                
            time.sleep(2)

def draw_highlighted_text(img, text, position, text_color, bg_color):
    font_scale, thickness = 0.7, 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - text_size[1] - 5), (x + text_size[0] + 4, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x + 2, y), font, font_scale, text_color, thickness)

# --- NEW: Spatial Math Helper ---
def is_same_position(box1, box2):
    """Checks if the center of a new box is very close to an old box."""
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    
    # Calculate distance between centers
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    # Allow movement radius based on the width of the person
    allowed_movement_radius = (box1[2] - box1[0]) * 0.6 
    
    return dist < allowed_movement_radius

class CameraWorker(threading.Thread):
    def __init__(self, cam_name, source, detector, reid_matcher, global_tracker):
        super().__init__()
        self.cam_name, self.source = cam_name, source 
        self.detector, self.reid_matcher, self.global_tracker = detector, reid_matcher, global_tracker
        self.logger = ImageLogger()
        self.identity_anchor, self.logged_ids, self.last_results = {}, set(), []
        
        # --- NEW: Spatial Memory Tracker ---
        self.spatial_memory = {} # Format: {p_id: (bbox, timestamp)}
        
        self.running, self.frame_count = True, 0
        self.fps_start_time, self.fps_frame_count, self.displayed_fps = 0, 0, 0

    def run(self):
        print(f"[INFO] Booting AI vision for {self.cam_name}...")
        self.stream = IPVideoStream(self.source).start()
        self.fps_start_time = time.time()

        while self.running:
            now = time.time()
            self.fps_frame_count += 1
            if now - self.fps_start_time >= 1.0:
                self.displayed_fps, self.fps_frame_count, self.fps_start_time = self.fps_frame_count, 0, now

            ret, frame = self.stream.read()
            if not ret or frame is None: time.sleep(0.01); continue
            
            self.frame_count += 1
            
            ai_frame = cv2.resize(frame, (640, 480))
            tracked_objs = self.detector.detect(ai_frame)
            scale_x, scale_y = frame.shape[1]/640, frame.shape[0]/480
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
            for obj in tracked_objs:
                x1, y1, x2, y2, conf, t_id = obj
                t_id = int(t_id) 
                bbox = [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]
                anchor_key = f"{self.cam_name}_{t_id}"
                
                p_id = self.identity_anchor.get(anchor_key)
                
                # --- NEW: Spatial Recovery if YOLO loses the track ---
                if p_id is None:
                    for old_p_id, (old_bbox, old_time) in list(self.spatial_memory.items()):
                        if now - old_time < 2.0: # Only trust memory less than 2 seconds old
                            if is_same_position(old_bbox, bbox):
                                p_id = old_p_id
                                self.identity_anchor[anchor_key] = p_id
                                # We don't want to print this every frame, but it's silently saving the ID!
                                break
                        else:
                            del self.spatial_memory[old_p_id]
                
                # Let the brain run if still unknown, or every 10 frames to learn
                if p_id is None or self.frame_count % 10 == 0:
                    new_p_id = self.reid_matcher.identify_and_register(frame, bbox, t_id, self.cam_name, existing_id=p_id)
                    if "Scanning" not in str(new_p_id):
                        p_id = new_p_id
                        self.identity_anchor[anchor_key] = p_id
                
                if p_id is None: p_id = "Scanning..."

                if "Scanning" not in str(p_id): 
                    self.global_tracker.mark_seen(p_id, self.cam_name) 
                    # Update Spatial Memory Coordinates
                    self.spatial_memory[p_id] = (bbox, now)
                
                if "Stranger" in str(p_id) and t_id not in self.logged_ids:
                    self.logger.log_stranger(frame, bbox, self.cam_name)
                    self.logged_ids.add(t_id)
                    
                self.last_results.append((bbox, p_id))

            for bbox, p_id in self.last_results:
                box_color = (0,255,0) if "Person" in str(p_id) else (0,165,255) if "Stranger" in str(p_id) else (0,255,255)
                bg_color = (0,100,0) if "Person" in str(p_id) else (0,80,200) if "Stranger" in str(p_id) else (0,150,150)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, 2)
                draw_highlighted_text(frame, str(p_id), (int(bbox[0]), int(bbox[1]) - 10), (255, 255, 255), bg_color)

            draw_highlighted_text(frame, f"FPS: {self.displayed_fps}", (15, 30), (255, 255, 255), (0, 0, 0))
            cv2.imshow(f"Local AI Feed: {self.cam_name}", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False
        
        self.stream.stop()

def main():
    print("[SYSTEM] Starting Local AI Surveillance...")
    with open("config.yaml", "r") as f: config = yaml.safe_load(f)
        
    db = DBHandler(config['database']['db_path'])
    detector = PersonDetector(config['models']['detection_weights'])
    reid_matcher = ReIDMatcher(db)
    
    global_tracker = GlobalPremiseTracker(db, reid_matcher)
    global_tracker.start()

    workers = []
    for name, source in config['cameras'].items():
        src = int(source) if str(source).isdigit() else source
        worker = CameraWorker(name, src, detector, reid_matcher, global_tracker)
        worker.start()
        workers.append(worker)

    print("\n[ACTIVE] System running locally. Press 'q' on the video window to safely exit.\n")
    
    try:
        while any(w.is_alive() for w in workers): time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Keyboard Interrupt received.")
    finally:
        print("\n[SYSTEM] Shutting down. Saving final data...")
        for w in workers: w.running = False
        global_tracker.running = False
        cv2.destroyAllWindows()
        db.close()
        print("[SYSTEM] Shutdown complete.")

if __name__ == "__main__": 
    main()