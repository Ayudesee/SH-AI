import tensorflow as tf
from imagegrab import grab_screen as gs
import numpy as np
import cv2
from directkeys import ReleaseKey, PressKey, A, D

vertices = np.array([[0, 8], [51, 8], [54, 0], [91, 0], [95, 11], [151, 11], [151, 103], [0, 103]])  # main screen coords



def main():
    filepath = 'models/model-10-13-21-59'

    model = tf.keras.models.load_model(filepath=filepath)
    while True:
        screen = gs()
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (152, 104))
        screen = cv2.Canny(screen, threshold1=140, threshold2=170, apertureSize=3)
        cv2.imshow('w', screen)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        screen = np.reshape(screen, (-1, 152, 104, 1))
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
