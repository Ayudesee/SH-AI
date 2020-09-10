import tensorflow as tf
import numpy as np
import cv2


file_name = 'D:/Ayudesee/Other/PyProj/pythonProject2/training_data_after_Canny.npy'

training_data = np.load(file_name, allow_pickle=True)

imgs = []
choices = []
for data in training_data:
    imgs.append(data[0])
    choices.append(data[1])




imgs = np.array(imgs)
choices = np.array(choices)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(104, 152)))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(imgs, choices, epochs=5)

img_test = np.reshape(imgs[80], (1, -1))
predict = model.predict(img_test)
print(predict)

