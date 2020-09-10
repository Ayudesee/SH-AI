from alexnet import alexnet, alexnet2
import numpy as np
import os
import tensorflow as tf

data = np.load('training_data_after_Canny.npy', allow_pickle=True)

model = alexnet2(152, 104, 0.5, 3)

model.predict(data[0][0])

print(1)