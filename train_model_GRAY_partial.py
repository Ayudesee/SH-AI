import tensorflow as tf
import numpy as np
import cv2
import time

t = time.localtime()
month, day, hrs, mins = t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min
file_name = 'raw_data_screen.npy'
# file_name2 = 'D:/Ayudesee/Other/PyProj/pythonProject2/training_data_GRAY_v2.npy'
modelname = 'models/model-{}-{}-{}-{}'.format(month, day, hrs, mins)
log_dir = 'D:/Ayudesee/Other/PyProj/pythonProject2/logs/{}-{}-{}-{}'.format(month, day, hrs, mins)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def train_model_GRAY_partial():
    EPOCHS = 10
    # MODEL_NAME = 'models/model-9-13-20-35'
    FILE_I_END = 260
    WIDTH = 152
    HEIGHT = 104
    WIDTH_CENTER = 31
    LOAD_MODEL = False

    if LOAD_MODEL:
        model = tf.keras.models.load_model(MODEL_NAME)
    else:
        model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Flatten(input_shape=(104, 152)))
        model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(2, 2), strides=2, padding="same", activation=tf.nn.relu,
                                         input_shape=(WIDTH, HEIGHT, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), padding="same", activation=tf.nn.relu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten(input_shape=(WIDTH, HEIGHT)))

        # model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # model.add(tf.keras.layers.Dropout(0.1))
        # model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    for e in range(EPOCHS):
        data_order = [i for i in range(1, FILE_I_END + 1)]
        data_partial = []
        data_for_evaluation = []
        for count, i in enumerate(data_order):
            try:
                file_name = 'D:/Ayudesee/Other/Data/processed_two_images/data{}.npy'.format(i)  # full file info
                train_data = np.load(file_name, allow_pickle=True)
                data_partial.extend(train_data)
                if i % 96 == 0:
                    print('training_data-{}.npy'.format(i), len(train_data))
                    data_partial = np.array(data_partial)
                    train = data_partial[:-len(train_data) // 10]
                    test = data_partial[-len(train_data) // 10:]

                    

                    X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
                    X = X / 255.0
                    Y = [i[1] for i in train]
                    X_test = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
                    X_test = X_test / 255.0
                    X_center_test = np.array([i[1] for i in test]).reshape(-1, WIDTH_CENTER, WIDTH_CENTER, 1)
                    Y_test = [i[2] for i in test]
                    X = np.reshape(X, (-1, 152, 104, 1))
                    Y = np.array(Y)
                    Y_test = np.array(Y_test)
                    data_for_evaluation.append(test)
                    print('EPOCH = {}'.format(e))
                    model.fit(X, Y, epochs=1)  # , callbacks=tensorboard_callback
                    data_partial = []
            except Exception as e:
                print(str(e))
    model.save(modelname)

    print('{} saved'.format(modelname))
    model.evaluate(X_test, Y_test)


train_model_GRAY_partial()
