import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time

file_name = 'D:/Ayudesee/Other/PyProj/pythonProject2/training_data_GRAY_v2.npy'
training_data = list(np.load(file_name, allow_pickle=True))

img_test = []
choice_test = []

for data in training_data[-len(training_data) // 10:]:
    img_test.append(data[0])
    choice_test.append(data[1])

img_test = np.asarray(img_test)
choice_test = np.asarray(choice_test)

model = tf.keras.models.load_model('model1')

predictions = model.predict(img_test)
print(predictions)
