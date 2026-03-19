import logging
import threading
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from scipy.spatial.distance import cosine
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ReIDMatcher:
    def __init__(self, db_handler, config=None):
        config = config or {}
        self.db = db_handler
        self.state_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.error_counters = {
            "signature_extraction_failures": 0,
            "matching_failures": 0,
            "db_write_failures": 0,
        }

        self.known_signatures = self.db.load_all_signatures()
        self.signature_centroids = {}
        self.similarity_threshold = float(config.get("similarity_threshold", 0.75))
        self.recovery_similarity_threshold = float(config.get("recovery_similarity_threshold", 0.82))
        self.learning_similarity_ceiling = float(config.get("learning_similarity_ceiling", 0.92))
        self.merge_similarity_threshold = float(config.get("merge_similarity_threshold", 0.88))
        self.shortlist_size = int(config.get("shortlist_size", 5))
        self.max_signatures_per_id = int(config.get("max_signatures_per_id", 20))
        self.min_crop_size = int(config.get("min_crop_size", 32))
        self.disappeared_tracks = {}
        self.max_wait_time = float(config.get("max_wait_time", 3.0))
        self.merged_ids = {}
        self.stranger_count = len([k for k in self.known_signatures.keys() if "Stranger" in k])

        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch, "dml") and torch.dml.is_available():
            self.device = "dml"
        else:
            self.device = "cpu"

        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._rebuild_centroids()

    def _record_error(self, key, message, **context):
        self.error_counters[key] += 1
        self.logger.error("%s | count=%s | context=%s", message, self.error_counters[key], context)

    def _refresh_centroid(self, person_id):
        signatures = self.known_signatures.get(person_id)
        if signatures is None or len(signatures) == 0:
            self.signature_centroids.pop(person_id, None)
            return

        centroid = np.mean(signatures, axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            self.signature_centroids.pop(person_id, None)
            return
        self.signature_centroids[person_id] = centroid / norm

    def _rebuild_centroids(self):
        for person_id in list(self.known_signatures.keys()):
            self._refresh_centroid(person_id)

    def _select_candidate_ids(self, current_sig):
        if not self.signature_centroids:
            return []

        scored = []
        for person_id, centroid in self.signature_centroids.items():
            sim = 1.0 - cosine(current_sig, centroid)
            if np.isnan(sim):
                continue
            scored.append((sim, person_id))

        scored.sort(reverse=True)
        return [person_id for _, person_id in scored[: self.shortlist_size]]

    def get_signature(self, frame, bbox):
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        if crop.shape[0] < self.min_crop_size or crop.shape[1] < self.min_crop_size:
            return None
        try:
            img_tensor = self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                features = self.model(img_tensor)
            vec = features.cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return None
            return vec / norm
        except Exception as exc:
            self._record_error("signature_extraction_failures", "signature extraction failed", error=str(exc))
            return None

    def run_merger_pass(self):
        with self.state_lock:
            if len(self.known_signatures) < 2:
                return

            ids = list(self.known_signatures.keys())
            merged_this_pass = set()

            for i, id_a in enumerate(ids):
                if id_a in merged_this_pass or id_a not in self.known_signatures:
                    continue
                for id_b in ids[i + 1 :]:
                    if id_b in merged_this_pass or id_b not in self.known_signatures:
                        continue

                    centroid_a = self.signature_centroids.get(id_a)
                    centroid_b = self.signature_centroids.get(id_b)
                    if centroid_a is None or centroid_b is None:
                        continue

                    centroid_sim = 1.0 - cosine(centroid_a, centroid_b)
                    if np.isnan(centroid_sim) or centroid_sim < self.merge_similarity_threshold:
                        continue

                    max_sim = 0
                    for sig_a in self.known_signatures[id_a]:
                        for sig_b in self.known_signatures[id_b]:
                            sim = 1.0 - cosine(sig_a, sig_b)
                            if sim > max_sim:
                                max_sim = sim

                    if max_sim > self.merge_similarity_threshold:
                        keep_id, merge_id = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                        print(f"\n[AI MERGER] 🧹 Fused Duplicate: {merge_id} is actually {keep_id} (Match: {max_sim*100:.1f}%)")
                        combined = np.vstack([self.known_signatures[keep_id], self.known_signatures[merge_id]])
                        if len(combined) > self.max_signatures_per_id:
                            combined = combined[-self.max_signatures_per_id :]

                        if not self.db.merge_visitors(keep_id, merge_id, combined):
                            self._record_error("db_write_failures", "merge_visitors failed", keep_id=keep_id, merge_id=merge_id)
                            continue

                        self.known_signatures[keep_id] = combined
                        del self.known_signatures[merge_id]
                        self._refresh_centroid(keep_id)
                        self._refresh_centroid(merge_id)
                        self.merged_ids[merge_id] = keep_id
                        merged_this_pass.add(merge_id)

    def identify_and_register(self, frame, bbox, track_id, cam_name, existing_id=None):
        current_sig = self.get_signature(frame, bbox)
        if current_sig is None:
            return existing_id or "Unknown"

        with self.state_lock:
            now = time.time()
            best_recovery_id, recovery_sim = None, 0
            for p_id, (d_time, d_sig) in list(self.disappeared_tracks.items()):
                if now - d_time > self.max_wait_time:
                    del self.disappeared_tracks[p_id]
                    continue
                sim = 1.0 - cosine(current_sig, d_sig)
                if sim > self.recovery_similarity_threshold and sim > recovery_sim:
                    recovery_sim, best_recovery_id = sim, p_id

            if best_recovery_id:
                del self.disappeared_tracks[best_recovery_id]
                return best_recovery_id

            best_match, max_sim = None, 0
            candidate_ids = self._select_candidate_ids(current_sig) or list(self.known_signatures.keys())
            for p_id in candidate_ids:
                for sig in self.known_signatures.get(p_id, []):
                    sim = 1.0 - cosine(current_sig, sig)
                    if sim > max_sim:
                        max_sim, best_match = sim, p_id

            if best_match and max_sim >= self.similarity_threshold:
                if max_sim < self.learning_similarity_ceiling and len(self.known_signatures[best_match]) < self.max_signatures_per_id:
                    print(f"[AI LEARNING] Learned pattern #{len(self.known_signatures[best_match]) + 1} for {best_match}!")
                    self.known_signatures[best_match] = np.vstack([self.known_signatures[best_match], current_sig])
                    self._refresh_centroid(best_match)
                    try:
                        self.db.update_signature(best_match, self.known_signatures[best_match])
                    except Exception as exc:
                        self._record_error("db_write_failures", "update_signature failed", person_id=best_match, error=str(exc))
                return best_match

            if existing_id is not None and ("Stranger" in existing_id or "Person" in existing_id):
                return existing_id

            self.stranger_count += 1
            final_id = f"Stranger_{self.stranger_count}"
            try:
                self.db.register_visitor(final_id, "Unverified", cam_name, "N/A", 300)
                new_sigs = np.array([current_sig], dtype=np.float32)
                self.db.save_signature(final_id, new_sigs)
                self.known_signatures[final_id] = new_sigs
                self._refresh_centroid(final_id)
            except Exception as exc:
                self._record_error("db_write_failures", "register/save signature failed", person_id=final_id, error=str(exc))
            return final_id

    def log_disappearance(self, person_id):
        with self.state_lock:
            if person_id in self.known_signatures:
                self.disappeared_tracks[person_id] = (time.time(), self.known_signatures[person_id][-1])

    def update_id_in_memory(self, old_id, new_id):
        with self.state_lock:
            if old_id not in self.known_signatures:
                self._record_error("matching_failures", "reassign id not present in memory", old_id=old_id, new_id=new_id)
                return

            self.known_signatures[new_id] = self.known_signatures.pop(old_id)
            self.merged_ids[old_id] = new_id
            self._refresh_centroid(new_id)
            self._refresh_centroid(old_id)
            if old_id in self.disappeared_tracks:
                self.disappeared_tracks[new_id] = self.disappeared_tracks.pop(old_id)
