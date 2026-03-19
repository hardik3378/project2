import threading
import time


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
