from ultralytics import YOLO
import cv2

# Charger un modèle plus rapide et léger (YOLOv5s, YOLOv8n, etc.)
model = YOLO('yolov8n.pt')  # ou 'yolov8n.pt' si tu veux tester v8

def detect_objects(frame):
    # Redimensionnement si nécessaire
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "label": label,
            "confidence": confidence,
            "box": [x1, y1, x2, y2]
        })

    return detections
