import numpy as np
import cv2

testpack1 = 'D:/Ayudesee/Other/Data/processed_two_images/data1.npy'
testpack2 = 'D:/Ayudesee/Other/Data/raw_data/raw_data_screen1.npy'
testpack3 = 'D:/Ayudesee/Other/Data/raw_data_processed_blurred/raw_data_screen_processed_blurred1.npy'
testpack4 = 'D:/Ayudesee/Other/Data/raw_data_shuffled/raw_data_screen_shuffled1.npy'

image1 = np.load(testpack1, allow_pickle=True)
image2 = np.load(testpack2, allow_pickle=True)
image3 = np.load(testpack3, allow_pickle=True)
image4 = np.load(testpack4, allow_pickle=True)
##print(image[0][0])

n = 100
cv2.imshow(testpack1, image1[n][0])
cv2.imshow('{}_2'.format(testpack1), image1[n][1])
cv2.imshow(testpack2, image2[n][0])
cv2.imshow(testpack3, image3[n][0])
cv2.imshow(testpack4, image4[n][0])
# print(np.array(image[n][1]).shape)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

