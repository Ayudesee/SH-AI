import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import time

train_data = np.load('training_data_GRAY.npy', allow_pickle=True)
print('train_data length = {}'.format(len(train_data)))
df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
nothings = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    img = cv2.Canny(img, threshold1=1, threshold2=1, apertureSize=3)
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
lefts = lefts[:len(rights)]
rights = rights[:len(lefts)]

final_data = nothings + rights + lefts

shuffle(final_data)

print('final_data length = {}'.format(len(final_data)))

np.save('training_data_after_Canny.npy', final_data)

for data in final_data:
    img = data[0]
    choice = data[1]
    img = cv2.resize(img, (768, 520))
    cv2.imshow('test', img)
    print(choice)
    time.sleep(0.1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
