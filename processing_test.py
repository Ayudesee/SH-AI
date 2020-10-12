import cv2
import numpy as np


pathdir = 'D:/Ayudesee/Other/Data/raw_data'

img = cv2.imread('D:/Ayudesee/Other/PyProj/SH-AI/imgs/img70.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('orig', img)
img_proc = cv2.Canny(img, threshold1=140, threshold2=170, apertureSize=3)
cv2.imshow('proc', img_proc)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
