import tensorflow as tf
import numpy as np
import cv2
import time

t = time.localtime()
month, day, hrs, mins = t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min
file_name = 'D:/Ayudesee/Other/PyProj/pythonProject2/training_data_BlackSndWhite_2_v2.npy'
file_name2 = 'D:/Ayudesee/Other/PyProj/pythonProject2/training_data_GRAY_v2.npy'
modelname = 'models/model-{}-{}-{}-{}'.format(month, day, hrs, mins)
log_dir = 'D:/Ayudesee/Other/PyProj/pythonProject2/logs/{}-{}-{}-{}'.format(month, day, hrs, mins)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def train_model():
    training_data = np.load(file_name, allow_pickle=True)
    training_data_2 = np.load(file_name2, allow_pickle=True)


    tf.keras.applications.inception_v3
    imgs = []
    choices = []
    imgs_test = []
    choices_test = []
    for data in training_data[:-100]:
        imgs.append(data[0])
        choices.append(data[1])
    for data in training_data[-100:]:
        imgs_test.append(data[0])
        choices_test.append(data[1])
    for data in training_data_2[:-100]:
        imgs.append(data[0])
        choices.append(data[1])
    for data in training_data_2[-100:]:
        imgs_test.append(data[0])
        choices_test.append(data[1])

    imgs = np.array(imgs, dtype='float32')
    imgs = np.reshape(imgs, (-1, 104, 152, 1))
    choices = np.array(choices)
    imgs_test = np.array(imgs_test)
    imgs_test = np.reshape(imgs_test, (-1, 104, 152, 1))
    choices_test = np.array(choices_test)

    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(104, 152)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(104, 152, 1), strides=(4, 4)))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten(input_shape=(104, 152)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    model.fit(imgs, choices, epochs=10, callbacks=tensorboard_callback)
    val_loss, val_acc = model.evaluate(imgs_test, choices_test)
    print('val_loss = {}, val_acc = {}'.format(val_loss, val_acc))
    model.save(modelname)


def load_and_fit(dataset):
    model = tf.keras.models.load_model(modelname)

    training_data = np.load(dataset, allow_pickle=True)
    imgs = []
    choices = []
    imgs_test = []
    choices_test = []
    for data in training_data[:-100]:
        imgs.append(data[0])
        choices.append(data[1])
    for data in training_data[-100:]:
        imgs_test.append(data[0])
        choices_test.append(data[1])
    imgs = np.array(imgs, dtype='uint8')
    imgs = np.reshape(imgs, (-1, 104, 152, 1))
    choices = np.array(choices)
    imgs_test = np.array(imgs_test)
    choices_test = np.array(choices_test)

    model.fit(imgs, choices, epochs=5, callbacks=tensorboard_callback)
    val_loss, val_acc = model.evaluate(imgs_test, choices_test)
    print('val_loss = {}, val_acc = {}'.format(val_loss, val_acc))
    model.save(modelname)


train_model()
# load_and_fit(file_name2)
