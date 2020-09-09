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


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_lines(img, lines):
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 1)
    except:
        pass


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=1, threshold2=1, apertureSize=3)
    # lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=15)
    # draw_lines(processed_img, lines)
    vertices = np.array([[0, 37], [310, 37], [330, 0], [430, 0], [460, 37],
                         [610, 37], [660, 50], [767, 50], [767, 519], [0, 519]])
    processed_img = roi(processed_img, [vertices])
    return processed_img


def main():
    file_name = 'C:/Users/Public/training_data_BlackSndWhite.npy'
    if os.path.isfile(file_name):
        print('File exists')
        training_data = list(np.load(file_name, allow_pickle=True))
    else:
        print('File does not exist, going from zero')
        training_data = []
    time.sleep(1)
    while True:

        screen = np.array(ImageGrab.grab(bbox=(0, 40, 768, 520)))
        keys = key_check()
        output = key_to_output(keys)
        # print('Loop took {} seconds'.format(time.time() - last_time))
        # last_time = time.time()
        screen = process_img(screen)
        cv2.imshow('window', screen)
        screen = cv2.resize(screen, (152, 104))

        training_data.append([screen, output])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)


main()
