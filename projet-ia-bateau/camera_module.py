from picamera2 import Picamera2
import cv2

class Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.start()

    def get_frame(self):
        frame = self.picam2.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
