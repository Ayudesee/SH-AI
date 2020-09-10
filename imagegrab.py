from PIL import ImageGrab
import numpy as np
import cv2


def grab_screen():
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 768, 520)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    cv2.imshow('window', screen)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    return screen
