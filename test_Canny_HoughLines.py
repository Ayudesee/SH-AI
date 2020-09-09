import numpy as np
from PIL import ImageGrab
import cv2
import time
import os


def draw_lines(img, lines):
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 1)
    except:
        pass


def process_img(original_image, threshold1, threshold2, apertureSize):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1, threshold2, apertureSize)
    # lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=15)
    # draw_lines(processed_img, lines)
    return processed_img


def main():
    count = 0
    threshold1 = 1
    threshold2 = 1
    apertureSize = 1
    time.sleep(2)
    while True:
        count += 1
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 768, 520)))
        # for i in range(10):
        #     for k in range(200):
        #         for j in range(200):
        #             threshold1 += 10
        #             screen = process_img(screen, threshold1, threshold2, apertureSize)
        #             filename = '/imgs/t1:{}, t2:{}, aS:{}.png'.format(threshold1, threshold2, apertureSize)
        #             cv2.imwrite(os.path.join(filename), screen)
        #         threshold2 += 10
        #     apertureSize += 1

        screen = process_img(screen, threshold1, threshold2, apertureSize)
        cv2.imshow('window', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    time.sleep(1)


main()
