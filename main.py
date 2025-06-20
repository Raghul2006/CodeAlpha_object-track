import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Download from https://github.com/abewley/sort

# Load YOLOv5 model
model = YOLO("yolov5s.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize SORT tracker
tracker = Sort()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame[..., ::-1])[0]
    detections = []
    for r in results.boxes:
        x1, y1, x2, y2 = r.xyxy[0].cpu().numpy()
        conf = float(r.conf)
        cls = int(r.cls)
        detections.append([x1, y1, x2, y2, conf])

    detections_np = np.array(detections)
    tracked_objects = tracker.update(detections_np)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Object Detection and Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
