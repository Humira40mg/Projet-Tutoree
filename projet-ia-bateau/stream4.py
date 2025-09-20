# stream4.py

from flask import Flask, Response
import cv2
from camera_module import Camera
from midas_module import estimate_depth
import numpy as np

app = Flask(__name__)
camera = Camera()

def gen_frames():
    while True:
        frame = camera.get_frame()

        # Estimation de la profondeur
        depth_map = estimate_depth(frame)

        # Normaliser pour visualiser
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        # Encode en JPEG
        ret, jpeg = cv2.imencode('.jpg', depth_vis)
        if not ret:
            continue

        # Envoie MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Flux vidéo avec estimation de profondeur (MiDaS)</h1><img src='/video_feed'>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
