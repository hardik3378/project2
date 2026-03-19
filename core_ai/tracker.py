from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)

    def update(self, detections, frame):
        deepsort_boxes = []
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            width = x2 - x1
            height = y2 - y1
            deepsort_boxes.append(([x1, y1, width, height], conf, 0))

        tracks = self.tracker.update_tracks(deepsort_boxes, frame=frame)

        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            detection_conf = getattr(track, "det_conf", 0.0) or 0.0
            tracked_objects.append([int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]), float(detection_conf), track_id])

        return tracked_objects
