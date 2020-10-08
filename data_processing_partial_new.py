import numpy as np
import cv2
import os

FILE_END = 260


def load_raw_data(i):
    filename_shuffled = 'D:/Ayudesee/Other/Data/raw_data_shuffled/raw_data_screen_shuffled{}.npy'.format(i)
    train_data = np.load(filename_shuffled, allow_pickle=True)
    return train_data


def roi(img, vertices1, vertices2):
    mask = np.full_like(img, fill_value=255)
    cv2.fillPoly(mask, vertices1, (0, 0, 0))
    cv2.fillPoly(mask, vertices2, (255, 255, 255))
    masked = cv2.bitwise_or(img, mask)
    return masked


def process_img(original_image):
    x1, x2, y1, y2 = 61, 91, 37, 67  # center coords
    vertices_screen = np.array([[0, 8], [51, 8], [54, 0], [91, 0], [95, 11], [151, 11], [151, 103], [0, 103]])  # main screen coords
    vertices_screen_center = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=1, threshold2=1, apertureSize=3)
    processed_img_center = processed_img[y1:y2+1, x1:x2+1]
    processed_img = roi(processed_img, [vertices_screen], [vertices_screen_center])
    return processed_img, processed_img_center


def main():
    for i in range(1, FILE_END + 1):
        full_processed_data = []
        filename_processed = 'D:/Ayudesee/Other/Data/processed_two_images/data{}.npy'.format(i)
        raw_data = load_raw_data(i)
        for data in raw_data:  # data[0] = [[img], [choice]]
            img, img_center = process_img(data[0])
            full_processed_data.append([img, img_center, data[1]])

        full_processed_data = np.array(full_processed_data)
        np.save(filename_processed, full_processed_data)
        print('Saved to: {}'.format(filename_processed))


main()
