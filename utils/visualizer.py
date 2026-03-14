import cv2

class Visualizer:
    def __init__(self):
        self.colors = {
            'safe': (100, 255, 100),    # Green
            'alert': (0, 0, 255),       # Red
            'stranger': (0, 165, 255),  # Orange
            'unknown': (150, 150, 150)  # Gray
        }

    def draw_corner_box(self, frame, bbox, color, length=20):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Draw the 8 lines for corners
        cv2.line(frame, (x1, y1), (x1+length, y1), color, 2)
        cv2.line(frame, (x1, y1), (x1, y1+length), color, 2)
        cv2.line(frame, (x2, y1), (x2-length, y1), color, 2)
        cv2.line(frame, (x2, y1), (x2, y1+length), color, 2)
        cv2.line(frame, (x1, y2), (x1+length, y2), color, 2)
        cv2.line(frame, (x1, y2), (x1, y2-length), color, 2)
        cv2.line(frame, (x2, y2), (x2-length, y2), color, 2)
        cv2.line(frame, (x2, y2), (x2, y2-length), color, 2)

    def draw_label(self, frame, bbox, text, color):
        x1, y1 = int(bbox[0]), int(bbox[1])
        # Background for text
        cv2.rectangle(frame, (x1, y1-20), (x1 + (len(text)*10), y1), (0,0,0), -1)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_hud(self, frame, registered_count, stranger_count):
        # Sleek Top-Left Dashboard
        cv2.rectangle(frame, (10, 10), (280, 80), (30, 30, 30), -1)
        cv2.putText(frame, "AI SECURITY MONITOR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Registered: {registered_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(frame, f"Strangers: {stranger_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)