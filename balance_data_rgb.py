import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import time


FILE_I_END = 737
counter_for_filename = 1
final_data = []

for i in range(1, FILE_I_END + 1):
    file_name = 'D:/Ayudesee/Other/Data/raw_data/raw_data_screen{}.npy'.format(i)
    file_name_shuffled = 'D:/Ayudesee/Other/Data/raw_data_shuffled/data{}.npy'.format(counter_for_filename)
    train_data = np.load(file_name, allow_pickle=True)
    lefts = []
    rights = []
    nothings = []
    shuffle(train_data)


    for data in train_data:
        img = data[0]
        choice = data[1]
        if choice == [0, 0, 1]:
            rights.append([img, choice])
        elif choice == [1, 0, 0]:
            lefts.append([img, choice])
        elif choice == [0, 1, 0]:
            nothings.append([img, choice])
        else:
            print('something wrong with data')

    nothings = nothings[:len(lefts)*2][:len(rights)*2]  # простоев в 2 раза больше движений (тестирование данных)
    # nothings = nothings[:-len(nothings) // 4]  # отсекаем 1/4 действий без движения (балансируем данные)
    lefts = lefts[:len(rights)]
    rights = rights[:len(lefts)]

    temp_data = nothings + rights + lefts
    for data in temp_data:
        final_data.append(data)

    print('final_data length = {}'.format(len(final_data)))
    if len(final_data) >= 500:
        counter_for_filename += 1
        np.save(file_name_shuffled, final_data[0:500])
        final_data = final_data[500:]