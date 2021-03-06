import numpy as np
import cv2
import os

FILE_END = 235


def load_raw_data(i):
    file_name_shuffled = 'D:/Ayudesee/Other/Data/raw_data_shuffled/data{}.npy'.format(i)
    train_data = np.load(file_name_shuffled, allow_pickle=True)
    return train_data


def roi(img, vertices):
    mask = np.full_like(img, fill_value=255)
    cv2.fillPoly(mask, vertices, (0, 0, 0))
    masked = cv2.bitwise_or(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=140, threshold2=170, apertureSize=3)
    vertices_screen = np.array([[0, 8], [51, 8], [54, 0], [91, 0], [95, 11], [151, 11], [151, 103], [0, 103]])  # main screen coords
    processed_img = roi(processed_img, [vertices_screen])
    return processed_img


def main():
    for i in range(1, FILE_END + 1):
        # imgs = []
        # choices = []
        full_processed_data = []
        file_name_processed = 'D:/Ayudesee/Other/Data/raw_data_shuffled_processed/data{}.npy'.format(i)
        raw_data = load_raw_data(i)
        for data in raw_data:  # data[0] = [[img], [choice]]
            full_processed_data.append([process_img(data[0]), data[1]])

        full_processed_data = np.array(full_processed_data)
        np.save(file_name_processed, full_processed_data)
        print('Saved to: {}'.format(file_name_processed))


main()
