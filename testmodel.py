import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import datetime

file_name = 'D:/Ayudesee/Other/PyProj/pythonProject2/training_data_after_Canny.npy'
# file_name = 'D:/Ayudesee/Other/PyProj/pythonProject2/training_data_GRAY.npy'

training_data = list(np.load(file_name, allow_pickle=True))
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

    # cv2.imshow('window', data[0])
    # print(data[1])
img = np.asarray(img)
choice = np.asarray(choice)
img_test = np.asarray(img_test)
choice_test = np.asarray(choice_test)

# img = np.resize(-1, 1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


log_dir = "D:/Ayudesee/Other/PyProj/pythonProject2/log"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.fit(img, choice, epochs=5, callbacks=[tensorboard_callback])

val_loss, val_acc = model.evaluate(img_test, choice_test)
print(val_loss, val_acc)

# model.save('D:/Ayudesee/Other/PyProj/pythonProject2/model1')
# print(img_test[0])
# predictions = model.predict(img_test)

# print(predictions)
