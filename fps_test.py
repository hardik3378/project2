import cv2
import time
from ultralytics import YOLO

# --- STRESS TEST CONFIG ---
# Change this string to test different models! 
# Try: "yolo12m.pt", "yolo12s.pt", "yolo11n.pt"
MODEL_TO_TEST = "yolo12n.pt" 
CAMERA_SOURCE = 0 # Use 0 for your laptop webcam, or paste your IP camera URL here

print(f"\n[INFO] Booting Stress Test for: {MODEL_TO_TEST}")
print("[INFO] (If this is your first time using this model, it will download now...)")

# Load the model
model = YOLO(MODEL_TO_TEST)

# Connect to camera
cap = cv2.VideoCapture(CAMERA_SOURCE)
prev_time = 0

print("\n[ACTIVE] Camera live. Look at the FPS in the top left corner.")
print("[ACTIVE] Press 'q' to quit the test.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[ERROR] Could not read from camera.")
        break

    # 1. Resize frame to match how your main app processes it
    test_frame = cv2.resize(frame, (640, 480))
    
    # 2. Run the heavy YOLO inference
    results = model(test_frame, classes=[0], verbose=False) # class 0 is person
    
    # 3. Calculate exact FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 4. Draw boxes and FPS on the screen
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Model: {MODEL_TO_TEST} | FPS: {int(fps)}", (15, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("YOLO Benchmark Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()