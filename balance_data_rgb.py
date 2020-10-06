import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import time


FILE_I_END = 578

for i in range(1, FILE_I_END + 1):
    file_name = 'C:/Users/Public/raw_data/raw_data_screen{}.npy'.format(i)
    file_name_shuffled = 'C:/Users/Public/raw_data_shuffled/raw_data_screen_shuffled{}.npy'.format(i)
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
            print('something wrong with choices')

    nothings = nothings[:len(lefts)][:len(rights)]
    nothings = nothings[:-len(nothings) // 4]  # отсекаем 1/4 действий без движения (балансируем данные)
    lefts = lefts[:len(rights)]
    rights = rights[:len(lefts)]

    final_data = nothings + rights + lefts

    # df = pd.DataFrame(final_data)
    # print(Counter(df[1].apply(str)))

    shuffle(final_data)

    print('final_data length = {}'.format(len(final_data)))

    np.save(file_name_shuffled, final_data)

# for data in final_data:
#     img = data[0]
#     choice = data[1]
#     img = cv2.resize(img, (768, 520))
#     cv2.imshow('test', img)
#     print(choice)
#     time.sleep(0.1)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
