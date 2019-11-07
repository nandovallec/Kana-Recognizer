import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

DATADIR = "D:\\Descargas Ant\\alcon2019\\dataset\\train\\imgs\\"
path = 'D:\\Descargas Ant\\alcon2019\\dataset\\train_kana\\'


CATEGORIES = []
#
# for category in CATEGORIES:  # do dogs and cats
#     path = os.path.join(DATADIR,category)  # create path to dogs and cats
#     for img in os.listdir(path):  # iterate over each image per dogs and cats
#         img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
#         plt.imshow(img_array, cmap='gray')  # graph it
#         plt.show()  # display!

NEW_HEIGTH = 60
NEW_WIDTH = 72
training_data = []
counter = 0
CONTRAST = 50
BRIGHTNESS = 20
TAGS = ['U+3042', 'U+3044', 'U+3046', 'U+3048', 'U+304A', 'U+304B', 'U+304D', 'U+304F', 'U+3051', 'U+3053', 'U+3055', 'U+3057', 'U+3059', 'U+305B', 'U+305D', 'U+305F', 'U+3061', 'U+3064', 'U+3066', 'U+3068', 'U+306A', 'U+306B', 'U+306C', 'U+306D', 'U+306E', 'U+306F', 'U+3072', 'U+3075', 'U+3078', 'U+307B', 'U+307E', 'U+307F', 'U+3080', 'U+3081', 'U+3082', 'U+3084', 'U+3086', 'U+3088', 'U+3089', 'U+308A', 'U+308B', 'U+308C', 'U+308D', 'U+308F', 'U+3090', 'U+3091', 'U+3092', 'U+3093']


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

for r, d, f in os.walk(path):
    for adding in d:
        for r1,d1,f1 in os.walk(path+adding+'\\'):
            for i in f1:
                #print(os.path.join(r1))
                counter+=1
                label = TAGS.index(adding)
                # if counter==3256:
                n = os.path.join(r1,i)
                #img_array = cv2.imread(n, cv2.IMREAD_GRAYSCALE)
                # img_array = img_array
                # imgContrast = cv2.pow(img_array/255.0, 1.1)
                # img_array = imgContrast
                #

                img = Image.open(n)
                img = img.convert('L')  # Gray scale
                n_img = ImageEnhance.Contrast(img).enhance(1.5)
                img = n_img.filter(ImageFilter.RankFilter(3, 2))
                img = img.resize((NEW_WIDTH, NEW_HEIGTH), Image.ANTIALIAS)

                new_array = np.asanyarray(img)
                # kernel = np.ones((4, 4), np.uint8)
                #img_array = adjust_gamma(img_array, 0.5)
                # imgMorph = cv2.erode(imgContrast, kernel, iterations=1)
                # img_array = imgMorph

                #new_array = cv2.resize(img_array, (NEW_WIDTH, NEW_HEIGTH))
                training_data.append([new_array, label])

                # plt.imshow(img_array, cmap='gray')  # graph it
                # plt.show()
                # plt.imshow(imgContrast, cmap='gray')  # graph it
                # plt.show()
                # plt.imshow(imgMorph, cmap='gray')  # graph it
                # plt.show()
                #
                # break

        # counter += 1
        print("Finish ",adding," one folder. ", counter," elements")
        CATEGORIES.append(adding)
        counter = 0
        # break
    #     if (counter == 11):
    #         break
    # break

print(CATEGORIES)
print(len(CATEGORIES))

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, NEW_WIDTH, NEW_HEIGTH, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()










