from pathlib import Path
import pandas as pd
from PIL import ImageFilter, ImageEnhance, Image
import numpy as np
import cv2
from keras import utils as np_utils
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder


# Utility functions
def unicode_to_kana(code: str):
    # Example: unicode_to_kana('U+304A')
    assert len(code) == 6
    return chr(int(code[2:], 16))


def unicode_to_kana_list(codes: list):
    # Example: unicode_to_kana_list( ['U+304A','U+304B','U+304D'] )
    assert len(codes) == 3
    return [unicode_to_kana(x) for x in codes]


def kana_to_unicode(kana: str):
    assert len(kana) == 1
    return 'U+' + hex(ord(kana))[2:]


def evaluation(y0, y1):
    cols = ['Unicode1', 'Unicode2', 'Unicode3']
    x = y0[cols] == y1[cols]
    x2 = x['Unicode1'] & x['Unicode2'] & x['Unicode3']
    acc = sum(x2) / len(x2) * 100
    # n_correct = (np.array(y0[cols]) == np.array(y1[cols])).sum()    #Regla de tres para calcular porcentaje
    # acc = n_correct / (len(y0)*3) * 100
    return acc


# This class manages all targets: train, val, and test
class AlconTargets():
    """
    This class load CSV files for train and test.
    It generates validation automatically.

    Arguments:
       + datapath is string designate path to the dataset, e.g., './dataset'
       + train_ratio is a parameter for amount of traindata.
         The remain will be the amount of validation.
    """

    def __init__(self, datapath: str, train_ratio: float):
        self.datapath = Path(datapath)

        # Train annotation
        fnm = Path(datapath) / Path('train') / 'annotations.csv'
        assert fnm.exists()
        df = pd.read_csv(fnm).sample(frac=1)  # Frac will randomize the rows

        # Split train and val
        nTrain = round(len(df) * train_ratio)
        self.train = df.iloc[0:nTrain]
        self.val = df.iloc[nTrain:]

        print("Num.Training: ", nTrain)

        # Test annotation
        fnm = Path(datapath) / Path('test') / 'annotations.csv'
        assert fnm.exists()
        self.test = pd.read_csv(fnm)


class AlconDataset():
    """
    This Dataset class provides an image and its unicodes.

    Arguments:
       + datapath is string designate path to the dataset, e.g., './dataset'
       + targets is DataFrame provided by AlconTargets, e.g., AlconTargets.train
       + isTrainVal is boolean variable.
    """

    def __init__(self, datapath: str, targets: 'DataFrame', isTrainVal: bool):
        # Targets
        self.targets = targets

        # Image path
        if isTrainVal:
            p = Path(datapath) / Path('train')
        else:
            p = Path(datapath) / Path('test')
        self.img_path = p / 'imgs'  # Path to images

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        # Get image
        ident = self.targets['ID'].iloc[idx]  # ID
        img_fnm = self.img_path / (str(ident) + '.jpg')  # image filename

        # img_fnm = self.img_path / (str(10000)+'.jpg')  # image filename

        assert img_fnm.exists()
        img = Image.open(img_fnm)
        # Get annotations
        # unicodes = list(self.targets.iloc[idx, 1:4])
        return img

    def showitem(self, idx: int):
        img, codes = self.__getitem__(idx)
        print(unicode_to_kana_list(codes))
        img.show()

    # You can fill out this sheet for submission
    def getSheet(self):
        sheet = self.targets.copy()  # Deep copy
        sheet[['Unicode1', 'Unicode2', 'Unicode3']] = None  # Initialization
        return sheet
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)




datapath='D:/Descargas Ant/alcon2019/dataset/'
targets = AlconTargets(datapath, train_ratio=0.8)
testdata = AlconDataset(datapath,targets.test, isTrainVal=False)
N = len(testdata)
sheet = testdata.getSheet()  # Get initial sheet
res_width = 40
res_height = 50
model = tf.keras.models.load_model("3.0T-128-2-128-1.model")


for i in range(N):
    # Prediction
    img = testdata[i]  # Get data
    # img.show()
    img = img.convert('L')  # Gray scale
    ne_img = ImageEnhance.Contrast(img).enhance(2)
    ers_img = ne_img.filter(ImageFilter.RankFilter(3, 1))
    res_img = ers_img.resize((res_width, res_height * 3), Image.ANTIALIAS)
    img = res_img
    # img.show()
    width, height = img.size
    area = (0, height / 3, width, height - height / 3)
    img2 = (img.crop(area))
    area = (0, (height / 3) * 2, width, height)
    img3 = (img.crop(area))
    area = (0, 0, width, height / 3)
    img = (img.crop(area))

    img = (np.array(img)).reshape(-1, res_width, res_height, 1)
    img2 = (np.array(img2)).reshape(-1, res_width, res_height, 1)
    img3 = (np.array(img3)).reshape(-1, res_width, res_height, 1)

    prediction = model.predict([img])



    y_pred1=np.argmax(model.predict([img])[0])
    y_pred2=np.argmax(model.predict([img2])[0])
    y_pred3=np.argmax(model.predict([img3])[0])

    y_pred1 = encoder.inverse_transform([y_pred1])
    y_pred2 = encoder.inverse_transform([y_pred2])
    y_pred3 = encoder.inverse_transform([y_pred3])

    # print(prediction)
    y_pred = [y_pred1[0], y_pred2[0], y_pred3[0]]
    #y_pred = unicode_to_kana_list(y_pred)
    # print(y_pred)
    # break
    if(i%1000 == 0):
        print('Prediction {}; {}; {}'.format(
            unicode_to_kana_list(y_pred),i,N))

    # Fill the sheet with y_pred
    sheet.iloc[i,1:4] = y_pred

sheet.to_csv('test_prediction.csv',index=False)

