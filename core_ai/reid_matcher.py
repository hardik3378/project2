import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from scipy.spatial.distance import cosine
import time

class ReIDMatcher:
    def __init__(self, db_handler):
        self.db = db_handler
        self.known_signatures = self.db.load_all_signatures()
        self.similarity_threshold = 0.75 
        self.disappeared_tracks = {} 
        self.max_wait_time = 3.0 
        self.merged_ids = {} 
        self.stranger_count = len([k for k in self.known_signatures.keys() if "Stranger" in k])

        if torch.cuda.is_available(): self.device = 'cuda'
        elif hasattr(torch, 'dml') and torch.dml.is_available(): self.device = 'dml'
        else: self.device = 'cpu'

        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier = torch.nn.Identity() 
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((256, 128)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_signature(self, frame, bbox):
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0: return None
        try:
            img_tensor = self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
            with torch.inference_mode(): features = self.model(img_tensor)
            vec = features.cpu().numpy().flatten()
            return vec / np.linalg.norm(vec)
        except Exception: return None

    def run_merger_pass(self):
        if len(self.known_signatures) < 2: return
        ids = list(self.known_signatures.keys())
        merged_this_pass = set()

        for i in range(len(ids)):
            id_a = ids[i]
            if id_a in merged_this_pass or id_a not in self.known_signatures: continue
            for j in range(i + 1, len(ids)):
                id_b = ids[j]
                if id_b in merged_this_pass or id_b not in self.known_signatures: continue

                max_sim = 0
                for sig_a in self.known_signatures[id_a]:
                    for sig_b in self.known_signatures[id_b]:
                        sim = 1.0 - cosine(sig_a, sig_b)
                        if sim > max_sim: max_sim = sim
                        
                if max_sim > 0.88: 
                    keep_id, merge_id = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                    print(f"\n[AI MERGER] 🧹 Fused Duplicate: {merge_id} is actually {keep_id} (Match: {max_sim*100:.1f}%)")
                    combined = np.vstack([self.known_signatures[keep_id], self.known_signatures[merge_id]])
                    if len(combined) > 20: combined = combined[-20:]
                    self.db.merge_visitors(keep_id, merge_id, combined)
                    self.known_signatures[keep_id] = combined
                    del self.known_signatures[merge_id]
                    self.merged_ids[merge_id] = keep_id
                    merged_this_pass.add(merge_id)

    # --- THE FIX: Added 'existing_id' to prevent ghost clones ---
    def identify_and_register(self, frame, bbox, track_id, cam_name, existing_id=None):
        current_sig = self.get_signature(frame, bbox)
        
        # If we can't extract an image, fall back to what YOLO already knows
        if current_sig is None: return existing_id or "Unknown"
        
        now = time.time()
        best_recovery_id, recovery_sim = None, 0
        for p_id, (d_time, d_sig) in list(self.disappeared_tracks.items()):
            if now - d_time > self.max_wait_time:
                del self.disappeared_tracks[p_id]; continue
            sim = 1.0 - cosine(current_sig, d_sig)
            if sim > 0.82 and sim > recovery_sim:
                recovery_sim, best_recovery_id = sim, p_id

        if best_recovery_id:
            del self.disappeared_tracks[best_recovery_id]
            return best_recovery_id

        best_match, max_sim = None, 0
        for p_id, stored_sigs in self.known_signatures.items():
            for sig in stored_sigs:
                sim = 1.0 - cosine(current_sig, sig)
                if sim > max_sim: max_sim, best_match = sim, p_id

        if max_sim >= self.similarity_threshold:
            if max_sim < 0.92 and len(self.known_signatures[best_match]) < 20:
                print(f"[AI LEARNING] Learned pattern #{len(self.known_signatures[best_match]) + 1} for {best_match}!")
                self.known_signatures[best_match] = np.vstack([self.known_signatures[best_match], current_sig])
                self.db.update_signature(best_match, self.known_signatures[best_match])
            return best_match
        else:
            # --- THE FIX: OCCLUSION ARMOR ---
            # The AI didn't recognize the face. BUT, if YOLO knows they are already here...
            if existing_id is not None and ("Stranger" in existing_id or "Person" in existing_id):
                # Trust YOLO. They just covered their face or turned away.
                return existing_id

            # If existing_id is None, it means they literally just walked into the camera frame
            self.stranger_count += 1
            final_id = f"Stranger_{self.stranger_count}"
            self.db.register_visitor(final_id, "Unverified", cam_name, "N/A", 300)
            
            new_sigs = np.array([current_sig])
            self.db.save_signature(final_id, new_sigs)
            self.known_signatures[final_id] = new_sigs
            return final_id

    def log_disappearance(self, person_id):
        if person_id in self.known_signatures: 
            self.disappeared_tracks[person_id] = (time.time(), self.known_signatures[person_id][-1])