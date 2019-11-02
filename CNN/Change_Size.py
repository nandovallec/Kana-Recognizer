import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import cv2
from PIL import ImageFilter, ImageEnhance, Image

path = 'D:\\Descargas Ant\\alcon2019\\dataset\\train_kana\\'
#files = [i for i in os.listdir("D:\\Descargas Ant\\alcon2019\\dataset\\rain_kana\\U+304A\\") if i.endswith("jpg")]
# x = []
# y = []
# sx = 0
# sy = 0
# counter = 0
# print("x is ", sx, " and y is ", sy)
#
# for r, d, f in os.walk(path):
#     for adding in d:
#         counter = 0
#         for r1,d1,f1 in os.walk(path+adding+'\\'):
#             for i in f1:
#                 #print(os.path.join(r1))
#                 counter+=1
#                 n = os.path.join(r1,i)
#                 img = Image.open(n)
#                 #x.append(img.size[0])
#                 #y.append(img.size[1])
#                 sx += img.size[0]
#                 sy += img.size[1]
#                 img.close()
#                 # if(counter%1000 == 0):
#                 #     print(counter)
#         print("Finish one folder. ", counter," elements")
#         x.append(sx/counter)
#         y.append(sy/counter)
#         print("average of ", sx/counter, " and ", sy/counter)
#         sx = 0
#         sy = 0
#


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)



# img_array = cv2.imread('./present/Contrast.jpg', cv2.IMREAD_GRAYSCALE)

# img_array = img_array
# imgContrast = cv2.pow(img_array/255.0, 1.1)
# img_array = imgContrast
#
# kernel = np.ones((2, 2), np.uint8)
# img_array = cv2.erode(img_array, kernel, iterations=1)
# img_array = adjust_gamma(img_array, 0.9)

img = Image.open("./Present/Contrast.jpg")
# img.show()
img = img.convert('L')  # Gray scale
img = img.filter(ImageFilter.RankFilter(3, 1))
img = ImageEnhance.Contrast(img).enhance(2)

img.save("./Present/Contrast4.jpg")



# cv2.imwrite('./present/Contrast2.jpg',img_array)
# print("x is ", sx, " and y is ", sy, "   counter ", counter)
# print("Sum of mean width is: ", sum(x))
# print("Sum of mean height is: ", sum(y))
# print("Mean width is: ", sum(x)/len(x))
# print("Mean height is: ", sum(y)/len(y))
# plt.plot(x, y, 'ro')
# #plt.show()