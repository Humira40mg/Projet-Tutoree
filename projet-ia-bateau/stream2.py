from flask import Flask, Response
import cv2
from camera_module import Camera  # ton module caméra
from object_detection import detect_objects

app = Flask(__name__)
camera = Camera()

def gen_frames():
    while True:
        frame = camera.get_frame()  # récupère image BGR np.array
        detections = detect_objects(frame)

        # Annoter les boîtes sur l'image
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            conf = det['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode l'image en JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Envoie en flux MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Flux vidéo avec détection YOLO</h1><img src='/video_feed'>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
