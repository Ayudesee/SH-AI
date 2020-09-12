import tensorflow as tf
from imagegrab import grab_screen_rgb as gs_rgb
import numpy as np
import cv2
from directkeys import ReleaseKey, PressKey, A, D


def main():
    filepath = 'models/model-9-12-12-58'

    model = tf.keras.models.load_model(filepath=filepath)
    while True:
        screen = gs_rgb()
        screen = np.array(screen)
        screen = cv2.resize(screen, (152, 104))
        screen = np.reshape(screen, (-1, 104, 152, 3))
        prediction = model.predict(screen)
        print(prediction)

        if np.argmax(prediction) == 0:
            PressKey(A)
            ReleaseKey(D)
        elif np.argmax(prediction) == 1:
            ReleaseKey(A)
            ReleaseKey(D)
        else:
            ReleaseKey(A)
            PressKey(D)


main()
