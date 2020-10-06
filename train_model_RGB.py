import tensorflow as tf
import numpy as np
import cv2
import time
# from models import inception_v3 as googlenet

t = time.localtime()
month, day, hrs, mins = t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min
file_name = 'raw_data_screen.npy'
# file_name2 = 'D:/Ayudesee/Other/PyProj/pythonProject2/training_data_GRAY_v2.npy'
modelname = 'models/model-{}-{}-{}-{}'.format(month, day, hrs, mins)
log_dir = 'D:/Ayudesee/Other/PyProj/pythonProject2/logs/{}-{}-{}-{}'.format(month, day, hrs, mins)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


# def train_inception():
#     training_data = np.load(file_name, allow_pickle=True)
#     test_data = np.load(file_name2, allow_pickle=True)
#
#     imgs = []
#     choices = []
#     imgs_test = []
#     choices_test = []
#     for data in training_data:
#         imgs.append(data[0])
#         choices.append(data[1])
#     for data in test_data:
#         imgs_test.append(data[0])
#         choices_test.append(data[1])
#     imgs = np.array(imgs, dtype='float32')
#     imgs = np.reshape(imgs, (-1, 104, 152, 1))
#     choices = np.array(choices)
#     imgs_test = np.array(imgs_test)
#     imgs_test = np.reshape(imgs_test, (-1, 104, 152, 1))
#     choices_test = np.array(choices_test)
#
#     base_model = InceptionV3(include_top=False, input_shape=(104, 152, 3))
#
#     add_model = tf.keras.models.Sequential()
#
#     add_model.add(tf.keras.layers.Dense(1024, activation='relu', input_shape=base_model.output_shape[1:]))
#     add_model.add(tf.keras.layers.Dropout(0.40))
#     add_model.add(tf.keras.layers.Flatten(input_shape=(104, 152)))
#     add_model.add(tf.keras.layers.Dense(3, activation='softmax'))
#     add_model.summary()
#
#     model = tf.keras.Model(inputs=base_model.input, outputs=add_model(base_model.output))
#     # opt = tf.keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     # reduce_lr = ReduceLROnPlateau(monitor='val_acc',
#     #                               patience=5,
#     #                               verbose=1,
#     #                               factor=0.1,
#     #                               cooldown=10,
#     #                               min_lr=0.00001)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
#     model.summary()
#     model.fit(imgs, choices, epochs=5, callbacks=tensorboard_callback)
#     val_loss, val_acc = model.evaluate(imgs_test, choices_test)
#     print('val_loss = {}, val_acc = {}'.format(val_loss, val_acc))
#     model.save(modelname)


def train_model_RGB():
    training_data = np.load(file_name, allow_pickle=True)
    # test_data = np.load(file_name2, allow_pickle=True)

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

    imgs = np.array(imgs, dtype='float32')
    imgs = np.reshape(imgs, (-1, 104, 152, 3))
    choices = np.array(choices)
    imgs_test = np.array(imgs_test)
    imgs_test = np.reshape(imgs_test, (-1, 104, 152, 3))
    choices_test = np.array(choices_test)

    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(104, 152)))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), padding="same", activation=tf.nn.relu, input_shape=(104, 152, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization(axis=1))

    # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    # model.add(tf.keras.layers.Dropout(0.4))
    # model.add(tf.keras.layers.BatchNormalization())

    # model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten(input_shape=(104, 152, 3)))

    # model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dropout(0.4))
    # model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
    model.add(tf.keras.layers.BatchNormalization())
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

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


def train_model_RGB_partial():
    EPOCHS = 10
    MODEL_NAME = 'models/model-9-13-20-35'
    FILE_I_END = 578
    WIDTH = 152
    HEIGHT = 104
    LOAD_MODEL = False
    # model = googlenet(WIDTH, HEIGHT, 3, LR, output=3, model_name='GN-1')

    if LOAD_MODEL:
        model = tf.keras.models.load_model(MODEL_NAME)
    else:
        model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Flatten(input_shape=(104, 152)))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=1, padding="same", activation=tf.nn.relu,
                                         input_shape=(WIDTH, HEIGHT, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1))
        model.add(tf.keras.layers.BatchNormalization(axis=1))

        # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        # model.add(tf.keras.layers.Dropout(0.1))
        # model.add(tf.keras.layers.BatchNormalization())

        # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        # model.add(tf.keras.layers.Dropout(0.1))
        # model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten(input_shape=(WIDTH, HEIGHT, 3)))

        # model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
        # model.add(tf.keras.layers.Dropout(0.4))
        # model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    for e in range(EPOCHS):
        data_order = [i for i in range(1, FILE_I_END + 1)]
        data20 = []
        data_for_evaluation = []
        for count, i in enumerate(data_order):
            try:
                file_name = 'C:/Users/Public/raw_data_shuffled/raw_data_screen_shuffled{}.npy'.format(i)
                # full file info
                train_data = list(np.load(file_name, allow_pickle=True))
                data20.extend(train_data)
                print('training_data-{}.npy'.format(i), len(train_data))
                if i % 578 == 0:
                    data20 = np.array(data20)
                    train = data20[:-len(train_data) // 10]
                    test = data20[-len(train_data) // 10:]

                    X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
                    Y = [i[1] for i in train]

                    X = np.reshape(X, (-1, 152, 104, 3))
                    Y = np.array(Y)
                    # data_for_evaluation.append(test)
                    print('EPOCH = {}'.format(e))
                    model.fit(X, Y, epochs=1, callbacks=tensorboard_callback)
                    data20 = []
            except Exception as e:
                print(str(e))
    model.save(modelname)
    print('{} saved'.format(modelname))
    # model.evaluate(data_for_evaluation[0], data_for_evaluation[1])



def train_model_RGB_test():
    training_data = np.load(file_name, allow_pickle=True)
    # test_data = np.load(file_name2, allow_pickle=True)

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

    imgs = np.array(imgs, dtype='float32')
    imgs = np.reshape(imgs, (-1, 152, 104, 3))
    choices = np.array(choices)
    imgs_test = np.array(imgs_test)
    imgs_test = np.reshape(imgs_test, (-1, 152, 104, 3))
    choices_test = np.array(choices_test)

    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(104, 152)))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, input_shape=(152, 104, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization(axis=1))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.BatchNormalization(axis=1))

    # model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten(input_shape=(152, 104, 3)))

    # model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dropout(0.4))
    # model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
    model.add(tf.keras.layers.BatchNormalization())
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    model.fit(imgs, choices, epochs=20, callbacks=tensorboard_callback)
    val_loss, val_acc = model.evaluate(imgs_test, choices_test)
    print('val_loss = {}, val_acc = {}'.format(val_loss, val_acc))
    model.save(modelname)


def train_model_GRAY_partial():
    EPOCHS = 20
    # MODEL_NAME = 'models/model-9-13-20-35'
    FILE_I_END = 578
    WIDTH = 152
    HEIGHT = 104
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
        data20 = []
        data_for_evaluation = []
        for count, i in enumerate(data_order):
            try:
                file_name = 'C:/Users/Public/raw_data_shuffled_processed/raw_data_screen_shuffled_processed{}.npy'.format(i)
                # full file info
                train_data = np.load(file_name, allow_pickle=True)
                data20.extend(train_data)
                if i % 96 == 0:
                    print('training_data-{}.npy'.format(i), len(train_data))
                    data20 = np.array(data20)
                    train = data20[:-len(train_data) // 10]

                    test = data20[-len(train_data) // 10:]

                    X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
                    X = X / 255.0
                    Y = [i[1] for i in train]
                    X_test = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
                    X_test = X_test / 255.0
                    Y_test = [i[1] for i in test]
                    X = np.reshape(X, (-1, 152, 104, 1))
                    Y = np.array(Y)
                    Y_test = np.array(Y_test)
                    data_for_evaluation.append(test)
                    print('EPOCH = {}'.format(e))
                    model.fit(X, Y, epochs=1, callbacks=tensorboard_callback)
                    data20 = []
            except Exception as e:
                print(str(e))
    model.save(modelname)

    print('{} saved'.format(modelname))
    model.evaluate(X_test, Y_test)


# train_inception()
# train_model_RGB_test()
train_model_GRAY_partial()
# load_and_fit(file_name2)
