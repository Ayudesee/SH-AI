import numpy as np
from PIL import ImageGrab
import tensorflow as tf
import cv2
import time


filepath = 'D:/Ayudesee/Other/PyProj/pythonProject2/model_after_Canny'


def process_image(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, 1, 1, 3)
    return processed_image


def main():
    model = tf.keras.models.load_model(filepath=filepath)
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 768, 520)))
    screen = cv2.resize(screen, (152, 104))
    screen = process_image(screen)
    screen = np.asarray(screen)


    for i in range(100):
        predictions = model.predict([screen])
        print(predictions)
        time.sleep(1)


if __name__ == '__main__':
    main()
