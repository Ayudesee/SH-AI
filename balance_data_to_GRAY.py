import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import time

train_data = np.load('raw_data_screen.npy', allow_pickle=True)
print('train_data length = {}'.format(len(train_data)))
df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
nothings = []

shuffle(train_data)


for data in train_data:
    img = apply_filters(data[0])
    choice = data[1]
    cv2.imshow('w', img)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    if choice == [0, 0, 1]:
        rights.append([img, choice])
    elif choice == [1, 0, 0]:
        lefts.append([img, choice])
    elif choice == [0, 1, 0]:
        nothings.append([img, choice])
    else:
        print('something wrong with data')





# nothings = nothings[:len(lefts)][:len(rights)]
# nothings = nothings[:-len(nothings) // 3]  # отсекаем 1/3 действий без движения (балансируем данные)
# lefts = lefts[:len(rights)]
# rights = rights[:len(lefts)]
#
# final_data = nothings + rights + lefts
#
# df = pd.DataFrame(final_data)
# print(Counter(df[1].apply(str)))
#
# shuffle(final_data)
#
# print('final_data length = {}'.format(len(final_data)))
#
# np.save('raw_data_screen_shuffled_to_GRAY.npy', final_data)






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
