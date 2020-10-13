from PIL import ImageGrab
import numpy as np
import cv2
import time


def grab_screen():
    screen = np.array(ImageGrab.grab(bbox=(0, 32, 768, 512)))
    # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    # screen = cv2.Canny(screen, threshold1=140, threshold2=170, apertureSize=3)
    return screen


def grab_screen_rgb():
    screen = np.array(ImageGrab.grab(bbox=(0, 32, 768, 512)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    return screen
