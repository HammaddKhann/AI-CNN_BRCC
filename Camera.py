import cv2
import pygame
import pygame.camera
from pygame.locals import *
import numpy as np


class Camera:

    def __init__(self):  # A constructor, Automatically called when instance of this class is created
        self.camera = cv2.VideoCapture(0)  # 0 -> The default camera on the system will be used

        if not self.camera.isOpened():  # If the camera is not found
            raise ValueError("Camera not found")

        self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)  # Width of the frames captured by the camera
        self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Height of the frames captured by the camera

    def get_frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            #frame = cv2.resize(frame, (120, 120))
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #frame.reshape(1, 120, 120, 3)

            if ret:  # If frame successfully read from the camera
                # Convert the frame to RGB format if it has only one channel
                if len(frame.shape) == 2:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return ret, frame_rgb
            else:
                return False, None
        else:
            return False, None

    def release(self):
        self.camera.release()

def main():
    camera = Camera()


if __name__ == "__main__":
    main()

