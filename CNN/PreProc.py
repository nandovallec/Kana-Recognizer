import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import os
import image_slicer
# read

i = 8569
path = "D:\\Descargas Ant\\alcon2019\\dataset\\train\\imgs\\"

img = Image.open('D:\\Descargas Ant\\alcon2019\\dataset\\train\\imgs\\'+str(i)+'.jpg')
img = img.convert('L')  # Gray scale
ne_img = ImageEnhance.Contrast(img).enhance(2)
img = ne_img.filter(ImageFilter.RankFilter(3, 1))

img.show()
res_img = img.resize((80,92*3),Image.ANTIALIAS)
img.save('orig.jpg')
res_img.save('resi.jpg')
img = res_img
width, height = img.size

area = (0,height/3, width, height - height/3)
img2 = (img.crop(area))
area = (0,(height/3)*2, width, height)
img3 = (img.crop(area))
area = (0,0, width, height/3)
img = (img.crop(area))


img.save('1.jpg')
img2.save('2.jpg')
img3.save('3.jpg')


# img.show()
# img2.show()
# img3.show()

# img3 = img











#
# img = cv2.imread(path+str(i)+'.jpg', cv2.IMREAD_GRAYSCALE)
# #img = cv2.imread('D:\\Descargas Ant\\alcon2019\\dataset\\train\\imgs\\1.jpg', cv2.IMREAD_GRAYSCALE)
#
# cv2.imwrite('.\\pruebas\\inp.png', img)
# img = cv2.resize(img,(90,400))
# # increase contrast
# pxmin = np.min(img)
# pxmax = np.max(img)
# imgContrast = ((img - pxmin) / (pxmax - pxmin)) * 350
#
# # increase line width
# kernel = np.ones((4, 4), np.uint8)
# imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)
# #imgMorph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#
# # write
# cv2.imwrite('.\\pruebas\\out.png', imgMorph)