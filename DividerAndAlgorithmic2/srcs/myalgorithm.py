###SECOND MAIN


import numpy as np
import cv2
import random
import pickle
from PIL import ImageFilter, ImageEnhance, Image

res_width = 40
res_height = 50
tam_feature_vec = 2048
class MyAlgorithm():
    """
    Build your algorithm.
    """
    def build_model(self,traindata):
        # Initialization
        num_train = len(traindata)
        print(num_train)



        model = np.zeros((num_train*3, tam_feature_vec))

        # We can use this part to divide the three letter images and then feed them the same way as the one letter training set
        #   we create the pickle set using this part of the code
        #   Uncomment the parts of the code that you need
        # TAGS = ['U+3042', 'U+3044', 'U+3046', 'U+3048', 'U+304A', 'U+304B', 'U+304D', 'U+304F', 'U+3051', 'U+3053',
        #         'U+3055', 'U+3057', 'U+3059', 'U+305B', 'U+305D', 'U+305F', 'U+3061', 'U+3064', 'U+3066', 'U+3068',
        #         'U+306A', 'U+306B', 'U+306C', 'U+306D', 'U+306E', 'U+306F', 'U+3072', 'U+3075', 'U+3078', 'U+307B',
        #         'U+307E', 'U+307F', 'U+3080', 'U+3081', 'U+3082', 'U+3084', 'U+3086', 'U+3088', 'U+3089', 'U+308A',
        #         'U+308B', 'U+308C', 'U+308D', 'U+308F', 'U+3090', 'U+3091', 'U+3092', 'U+3093']
        # training_data = []

        y_train = [ ['U+0000'] for i in range(num_train*3) ]




        # Convert images to features
        for i in range(num_train):
            img, codes = traindata[i]  # Get an image and label

            # img = np.array(img.convert('L'))  # Gray scale
            img = img.convert('L') # Gray scale


            ne_img = ImageEnhance.Contrast(img).enhance(2)
            ers_img = ne_img.filter(ImageFilter.RankFilter(3, 1))
            res_img = ers_img.resize((res_width, res_height * 3), Image.ANTIALIAS)
            img = res_img
            width, height = img.size
            area = (0, height / 3, width, height - height / 3)
            img2 = (img.crop(area))
            area = (0, (height / 3) * 2, width, height)
            img3 = (img.crop(area))
            area = (0, 0, width, height / 3)
            img = (img.crop(area))

            # img.show()
            # img2.show()
            # img3.show()



            img = np.array(img)
            img2 = np.array(img2)
            img3 = np.array(img3)



            ###########    If you need to create pickle
            # training_data.append([img,codes[0]])
            # training_data.append([img2, codes[1]])
            # training_data.append([img3, codes[2]])
            ###########


            # opencvImage = cv2.cvtColor(np.array(img),cv2.IMREAD_GRAYSCALE)
            # pxmin = np.min(opencvImage)
            # pxmax = np.max(opencvImage)
            # imgContrast = ((opencvImage - pxmin) / (pxmax - pxmin)) * 500
            #
            # kernel = np.ones((3, 3), np.uint8)
            # imgMorph = cv2.erode(imgContrast, kernel, iterations=2)
            # # imgMorph = np.array(imgMorph, dtype=float)/float(255)
            # imgMorph = imgMorph.astype(np.uint)
            # cv2.imwrite('out2.png', imgMorph)
            # cv2.imshow('image', imgMorph)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()



            # model[i,:] = np.resize(img,512)  # feature vector

            model[i, :] = np.resize(img, tam_feature_vec)
            model[i+num_train, :] = np.resize(img2, tam_feature_vec)
            model[i+num_train*2, :] = np.resize(img3, tam_feature_vec)

            #print(np.array(img).shape)
            #print(len(model[i,:]))
            #print(len(np.resize(img,512)))
            # print(codes)
            # print("meee"+codes[2])
            y_train[i] = codes[0]
            y_train[i+num_train] = codes[1]
            y_train[i+num_train*2] = codes[2]

            if i%1000 == 0:
                print("Training ", i)


            # break



        ########################################################################

        #In case you need to build the pickle data set
        # Shuffle the data just in case the words are ordered alphabetically
        # random.shuffle(training_data)
        #
        # X = []
        # y = []
        #
        # for features, label in training_data:
        #     X.append(features)
        #     y.append(label)
        #
        # X = np.array(X).reshape(-1, res_width, res_height, 1)
        #
        # pickle_out = open("X_THREE_2.pickle", "wb")
        # pickle.dump(X, pickle_out)
        # pickle_out.close()
        #
        # pickle_out = open("y_THREE_2.pickle", "wb")
        # pickle.dump(y, pickle_out)
        # pickle_out.close()

        ##########################################################################

        # Keep model and labels
        self.model = model
        self.y_train = y_train

    # Output is expected as list, ['U+304A','U+304A','U+304A']
    def predict(self,img):
        # img = np.array(img.convert('L'))

        img = img.convert('L')  # Gray scale
        ne_img = ImageEnhance.Contrast(img).enhance(2)
        ers_img = ne_img.filter(ImageFilter.RankFilter(3, 1))


        res_img = ers_img.resize((res_width, res_height * 3), Image.ANTIALIAS)
        img = res_img


        # We divide the test image in 3
        # We could also try to test with the testing set of 1 character images
        width, height = img.size
        area = (0, height / 3, width, height - height / 3)
        img2 = (img.crop(area))
        area = (0, (height / 3) * 2, width, height)
        img3 = (img.crop(area))
        area = (0, 0, width, height / 3)
        img = (img.crop(area))

        img = np.array(img)
        img2 = np.array(img2)
        img3 = np.array(img3)

        feat = np.resize(img,tam_feature_vec)  # feature vector
        feat2 = np.resize(img2,tam_feature_vec)  # feature vector
        feat3 = np.resize(img3,tam_feature_vec)  # feature vector

        # d1 = np.linalg.norm(self.model - feat, axis=1)  # measure distance --- Frobenius Norm / Euclidean norm
        dist = (np.sqrt(np.cumsum(np.square(self.model-feat), axis=1)))[:, -1]

        # d2 = np.linalg.norm(self.model - feat2, axis=1)  # measure distance --- Frobenius Norm / Euclidean norm
        dist2 = (np.sqrt(np.cumsum(np.square(self.model-feat2), axis=1)))[:, -1]

        # d3 = np.linalg.norm(self.model - feat3, axis=1)  # measure distance --- Frobenius Norm / Euclidean norm
        dist3 = (np.sqrt(np.cumsum(np.square(self.model-feat3), axis=1)))[:, -1]



        y_pred = self.y_train[ np.argmin(dist) ]  # Get the closest
        y_pred2 = self.y_train[ np.argmin(dist2) ]  # Get the closest
        y_pred3 = self.y_train[ np.argmin(dist3) ]  # Get the closest


        final_pred = [y_pred, y_pred2, y_pred3]

        return final_pred
