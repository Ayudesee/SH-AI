import cv2
import numpy as np


pathdir = 'D:/Ayudesee/Other/Data/raw_data'
savedir = 'D:/Ayudesee/Other/PyProj/SH-AI/imgs'
FILE_END = 737


def data2img(path, file_end):
    picnum = 0
    for i in range(1, file_end + 1):
        data = np.load('{}/raw_data_screen{}.npy'.format(path, i), allow_pickle=True)
        print('saving to {}/raw_data_screen{}.npy'.format(path, i))
        for k in data:
            picnum += 1
            cv2.imwrite(filename='imgs_gray/img{}.png'.format(picnum), img=cv2.cvtColor(k[0], cv2.COLOR_RGB2GRAY))


data2img(pathdir, FILE_END)
