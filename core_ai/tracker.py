from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonTracker:
    def __init__(self, max_age=30):
        """
        max_age: How many frames the tracker will "remember" a lost person 
        before assigning them a completely new ID. Good for when people walk behind pillars.
        """
        self.tracker = DeepSort(max_age=max_age)

    def update(self, boxes, frame):
        """
        Takes YOLO bounding boxes and updates their movement tracks.
        """
        # DeepSORT expects a specific format: [ [left, top, width, height], confidence, class ]
        deepsort_boxes = []
        for box in boxes:
            x1, y1, x2, y2, conf = box
            width = x2 - x1
            height = y2 - y1
            # Class '0' is person
            deepsort_boxes.append(([x1, y1, width, height], conf, 0)) 

        # Update the tracker with the current frame's detections
        tracks = self.tracker.update_tracks(deepsort_boxes, frame=frame)
        
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            # This is the temporary ID for the person on THIS specific camera
            track_id = track.track_id 
            
            # Get the coordinates back in Left, Top, Right, Bottom format
            ltrb = track.to_ltrb() 
            tracked_objects.append([int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]), track_id])
            
        return tracked_objects