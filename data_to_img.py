import cv2
import numpy as np


pathdir = 'D:/Ayudesee/Other/Data/raw_data'
# savedir = 'D:/Ayudesee/Other/PyProj/SH-AI/imgs_gray_processed'
FILE_END = 737


def roi(img, vertices1):
    mask = np.full_like(img, fill_value=255)
    cv2.fillPoly(mask, vertices1, (0, 0, 0))
    masked = cv2.bitwise_or(img, mask)
    return masked


def process_img(original_image):
    # x1, x2, y1, y2 = 61, 91, 37, 67  # center coords
    vertices_screen = np.array([[0, 8], [51, 8], [54, 0], [91, 0], [95, 11], [151, 11], [151, 103], [0, 103]])  # main screen coords

    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=140, threshold2=170, apertureSize=3)
    processed_img = roi(processed_img, [vertices_screen])
    return processed_img


def data2img(path, file_end):
    picnum = 0
    for i in range(1, file_end + 1):
        file = np.load('{}/raw_data_screen{}.npy'.format(path, i), allow_pickle=True)
        print('saving from {}/raw_data_screen{}.npy'.format(path, i))
        for data in file:
            picnum += 1
            img = process_img(data[0])
            cv2.imwrite(filename='imgs_gray_processed/img{}.png'.format(picnum), img=img)


data2img(pathdir, FILE_END)
