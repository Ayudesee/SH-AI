import numpy as np
import cv2
import os

file_name = 'C:/Users/Public/raw_data_shuffled/raw_data_screen_shuffled1.npy'


def load_raw_data():
    # if os.path.isfile(file_name):
    #     print('File exists')
    #     raw_data = np.load(file_name, allow_pickle=True)
    #     return raw_data
    # else:
    #     print('File does not found')
    FILE_I_END = 578

    for i in range(1, FILE_I_END + 1):
        # file_name = 'C:/Users/Public/raw_data/raw_data_screen{}.npy'.format(i)
        file_name_shuffled = 'C:/Users/Public/raw_data_shuffled/raw_data_screen_shuffled{}.npy'.format(i)
        train_data = np.load(file_name_shuffled, allow_pickle=True)
        return train_data



def draw_lines(img, lines):
    try:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), 255, 10)
    except:
        pass


def roi(img, vertices):
    mask = np.full_like(img, fill_value=255)
    cv2.fillPoly(mask, vertices, (0, 0, 0))
    masked = cv2.bitwise_or(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # processed_img = cv2.resize(processed_img, (768, 512))
    processed_img = cv2.Canny(processed_img, threshold1=1, threshold2=1, apertureSize=3)
    lines = cv2.HoughLinesP(processed_img, rho=1, theta=np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)
    draw_lines(processed_img, lines)


    vertices_screen_768_512 = np.array([[0, 37], [310, 37], [330, 0], [430, 0], [460, 37],
                                [610, 37], [640, 55], [767, 55], [767, 519], [0, 519]])
    vertices_screen = np.array([[0, 8], [51, 8], [54, 0], [91, 0], [95, 10], [151, 10], [151, 103], [0, 103]])
    processed_img = roi(processed_img, [vertices_screen])
    processed_img = cv2.resize(processed_img, (152, 104))
    return processed_img





def main():
    imgs = []
    choices = []
    raw_data = load_raw_data()
    for data in raw_data:
        imgs.append(data[0])
        choices.append((data[1]))
        # cv2.imread(data[0])
        data[0] = process_img(data[0])

        cv2.imshow('w', data[0])
        if cv2.waitKey(200) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
