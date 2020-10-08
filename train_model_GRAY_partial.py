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