"""
Press Q any time to exit.

Example:
python demo_webcam.py --resolution 1280 720 --hide-fps
"""

import argparse, os, time
import numpy as np
import cv2

from threading import Thread, Lock
from PIL import Image
import pyvirtualcam # pip install pyvirtualcam

# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Virtual webcam demo')

parser.add_argument('--hide-fps', action='store_true')
parser.add_argument('--resolution', type=int, nargs=2, metavar=('width', 'height'), default=(1920, 1080))
parser.add_argument('--camera-device', type=str, default='/dev/video0')
args = parser.parse_args()


# ----------- Utility classes -------------


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame
    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()

# An FPS tracker that computes exponentialy moving average FPS
class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio
    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + (1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()
    def get(self):
        return self._avg_fps

# Wrapper for playing a stream with cv2.imshow(). It can accept an image and return keypress info for basic interactivity.
# It also tracks FPS and optionally overlays info onto the stream.
class Displayer:
    def __init__(self, title, width=None, height=None, show_info=True):
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        self.webcam = None
        self.alpha =  255 * np.ones((self.height, self.width))
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)
    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            cv2.putText(image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        if self.webcam is not None:
            image_web = np.ascontiguousarray(image, dtype=np.uint8) # .copy()
            image_web = cv2.cvtColor(image_web, cv2.COLOR_RGB2BGR)
            image_web = np.dstack((image_web, self.alpha)).astype(np.uint8)
            self.webcam.send(image_web)
        # else:
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF


# --------------- Main ---------------

width, height = args.resolution
cam = Camera(width=width, height=height)

flat_w = 2 * cam.width
flat_h = cam.height // 2
half_flat_w = flat_w // 2

cropped_h = flat_h
cropped_w = (16 * flat_h) // 9

offset = flat_w // 4 - cropped_w // 2
rotation = 10


flat_frame = np.zeros((flat_h, flat_w, 3), dtype=np.uint8)
cropped_frame = np.zeros((cropped_h, cropped_w, 3), dtype=np.uint8)

dsp = Displayer('Insta 360 ONE X2 Webcam Preview', cropped_w, cropped_h, show_info=(not args.hide_fps))

dsp.webcam = pyvirtualcam.Camera(width=cropped_w, height=cropped_h, fps=20)

def rotateLeft():
    global offset
    offset = (offset - rotation) % (flat_w - cropped_w)
    if offset > (flat_w // 2 - cropped_w) and offset < flat_w // 2:
        offset = flat_w // 2 - cropped_w

def rotateRight():
    global offset
    offset = (offset + rotation) % (flat_w - cropped_w)
    if offset > (flat_w // 2 - cropped_w) and offset < flat_w // 2:
        offset = flat_w // 2

def app_step():
    global offset
    cropped_frame[:,:,:] = flat_frame[:, offset:offset+cropped_w, :]

    key = dsp.step(cropped_frame)

    if key == ord('a'):
        rotateLeft()
    if key == ord('d'):
        rotateRight()
    if key == ord('q'):
        return True

show_front = True
show_back = True
while True:
    frame = cam.read()
    flat_frame[0:flat_h, 0:half_flat_w, :] = frame[0:flat_h, :, :]
    flat_frame[0:flat_h, half_flat_w:flat_w, :] = frame[flat_h:frame.shape[0], :, :]
    if app_step():
        break
