import numpy as np
import cv2
import os

FILE_END = 578


def load_raw_data(i):
    file_name_shuffled = 'C:/Users/Public/raw_data_shuffled/raw_data_screen_shuffled{}.npy'.format(i)
    train_data = np.load(file_name_shuffled, allow_pickle=True)
    return train_data


def roi(img, vertices):
    mask = np.full_like(img, fill_value=255)
    cv2.fillPoly(mask, vertices, (0, 0, 0))
    masked = cv2.bitwise_or(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=1, threshold2=1, apertureSize=3)
    # lines = cv2.HoughLinesP(processed_img, rho=1, theta=np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)
    # draw_lines(processed_img, lines)
    # vertices_screen_768_512 = np.array([[0, 37], [310, 37], [330, 0], [430, 0], [460, 37],
    #                             [610, 37], [640, 55], [767, 55], [767, 519], [0, 519]])
    vertices_screen = np.array([[0, 8], [51, 8], [54, 0], [91, 0], [95, 10], [151, 10], [151, 103], [0, 103]])
    processed_img = roi(processed_img, [vertices_screen])
    processed_img = cv2.resize(processed_img, (152, 104))
    return processed_img


def main():
    for i in range(1, FILE_END + 1):
        imgs = []
        choices = []
        full_processed_data = []
        file_name_processed = 'C:/Users/Public/raw_data_shuffled_processed/raw_data_screen_shuffled_processed{}.npy'.format(i)
        raw_data = load_raw_data(i)
        for data in raw_data:  # data[0] = [[img], [choice]]
            # imgs.append(process_img(data[0]))
            # choices.append((data[1]))
            full_processed_data.append([process_img(data[0]), data[1]])

        full_processed_data = np.array(full_processed_data)
        np.save(file_name_processed, full_processed_data)
        print('Saved to: {}'.format(file_name_processed))


main()
