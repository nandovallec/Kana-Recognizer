import numpy as np
import cv2
from PIL import ImageFilter, ImageEnhance, Image

tam_feature_vec = 1024
class MyAlgorithm():
    """
    Build your algorithm.
    """
    def build_model(self,traindata):
        # Initialization
        num_train = len(traindata)
        print(num_train)
        model = np.zeros((num_train,tam_feature_vec))
        y_train = [ ['U+0000']*3 for i in range(num_train) ]
        
        # Convert images to features
        for i in range(num_train):
            img, codes = traindata[i]  # Get an image and label

            # img = np.array(img.convert('L'))  # Gray scale
            img = img.convert('L') # Gray scale
            ne_img = ImageEnhance.Contrast(img).enhance(2)
            ers_img = ne_img.filter(ImageFilter.RankFilter(3, 1))
            img = np.array(ers_img)



            # opencvImage = cv2.cvtColor(np.array(img),cv2.IMREAD_GRAYSCALE)
            # pxmin = np.min(opencvImage)
            # pxmax = np.max(opencvImage)
            # imgContrast = ((opencvImage - pxmin) / (pxmax - pxmin)) * 500
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

            y_train[i] = codes

            
        # Keep model and labels    
        self.model = model
        self.y_train = y_train

    # Output is expected as list, ['U+304A','U+304A','U+304A']
    def predict(self,img):
        # img = np.array(img.convert('L'))

        img = img.convert('L')  # Gray scale
        ne_img = ImageEnhance.Contrast(img).enhance(2)
        ers_img = ne_img.filter(ImageFilter.RankFilter(3, 1))
        img = np.array(ers_img)

        feat = np.resize(img,tam_feature_vec)  # feature vector


        dist = np.linalg.norm(self.model - feat, axis=1)  # measure distance --- Frobenius Norm / Euclidean norm
        # dist = (np.sum(np.abs(self.model - feat) ** 2, axis=-1) ** (1. / 2))         SLOWER

        y_pred = self.y_train[ np.argmin(dist) ]  # Get the closest
        return y_pred
