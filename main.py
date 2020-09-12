import numpy as np
from PIL import ImageGrab
import cv2
import ctypes
import time
import os
from keys import PressKey, ReleaseKey, keyA, keyD
import pyautogui as pgui
from getkeys import key_check


def key_to_output(keys):
    # [A, ,D]
    output = [0, 0, 0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    elif ' ':
        output[1] = 1

    return output


def start_recording():
    file_name = 'C:/Users/Public/raw_data_screen.npy'
    if os.path.isfile(file_name):
        print('File exists')
        training_data = list(np.load(file_name, allow_pickle=True))
    else:
        print('File does not exist, going from zero')
        training_data = []
    time.sleep(2)

    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 32, 768, 512)))
        keys = key_check()
        output = key_to_output(keys)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        # cv2.imshow('window', screen)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
        screen = cv2.resize(screen, (152, 104))
        training_data.append([screen, output])

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)


def main():
    start_recording()


main()
