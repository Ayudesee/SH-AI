import numpy as np
from alexnet import alexnet

WIDTH = 152
HEIGHT = 104
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'hexagon-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('training_data_GRAY_v2.npy')

train = train_data[:-160]
test = train_data[-160:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[0] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = np.array([i[0] for i in test])

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=160, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)