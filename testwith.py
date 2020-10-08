import numpy as np
import cv2

testpack = 'D:/Ayudesee/Other/Data/raw_data_processed_blurred/raw_data_screen_processed_blurred1.npy'

image = np.load(testpack, allow_pickle=True)
print(image[0][0])

cv2.imshow('w', image[5][0])
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

