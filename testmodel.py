import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import datetime

file_name = 'D:/Ayudesee/Other/PyProj/pythonProject2/training_data_after_Canny.npy'
# file_name = 'D:/Ayudesee/Other/PyProj/pythonProject2/training_data_GRAY.npy'

training_data = np.load(file_name, allow_pickle=True)
# plt.imshow(training_data[0])
img = []
choice = []
img_test = []
choice_test = []


for data in training_data[:-len(training_data) // 20]:
    img.append(data[0])
    choice.append(data[1])

for data in training_data[-len(training_data) // 20:]:
    img_test.append(data[0])
    choice_test.append(data[1])



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(247, 3, activation=tf.nn.relu, input_shape=(4, 104, 152, 3)))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = "D:/Ayudesee/Other/PyProj/pythonProject2/log"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
