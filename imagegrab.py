from PIL import ImageGrab
import numpy as np
import cv2
import time


def grab_screen():
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 768, 520)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    screen = cv2.Canny(screen, threshold1=1, threshold2=1, apertureSize=3)
    # cv2.imshow('window', screen)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    return screen
