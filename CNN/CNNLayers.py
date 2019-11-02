import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
from keras import utils as np_utils
from keras import backend
import time
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

import pickle


###tensorboard --logdir=logs/ --host localhost --port 8088
from tensorflow.python.keras.backend import set_session

dense_layers = [1]
layer_sizes = [128]
layer_sizes_conv = [128]
conv_layers = [2]



pickle_in = open("X_THREE_2.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y_THREE_2.pickle","rb")
y = pickle.load(pickle_in)
# TAGS = ['U+3042', 'U+3044', 'U+3046', 'U+3048', 'U+304A', 'U+304B', 'U+304D', 'U+304F', 'U+3051', 'U+3053', 'U+3055', 'U+3057', 'U+3059', 'U+305B', 'U+305D', 'U+305F', 'U+3061', 'U+3064', 'U+3066', 'U+3068', 'U+306A', 'U+306B', 'U+306C', 'U+306D', 'U+306E', 'U+306F', 'U+3072', 'U+3075', 'U+3078', 'U+307B', 'U+307E', 'U+307F', 'U+3080', 'U+3081', 'U+3082', 'U+3084', 'U+3086', 'U+3088', 'U+3089', 'U+308A', 'U+308B', 'U+308C', 'U+308D', 'U+308F', 'U+3090', 'U+3091', 'U+3092', 'U+3093']
# new_y = []
# for tag in y:
#     new_y.append(TAGS.index(tag))
#
# # print(y)
#
# #y = np.array(y)
# y = new_y
# # print(y)
X = X/255.0

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)
# w = y[0]
# print (np.shape(w))
# w = np.argmax(w)
# print(np.shape(w))
# w = encoder.inverse_transform([w])
# print(np.shape(w))
# # print(np.shape(y))



gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
backend.tensorflow_backend.set_session(sess)


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            for layer_size_conv in layer_sizes_conv:
                # if (conv_layer == 1 and layer_size_conv == 128 and dense_layer == 1 and layer_size == 64)\
                #     or (conv_layer == 1 and layer_size_conv == 256 and dense_layer == 1 and layer_size == 64)\
                #     or (conv_layer == 1 and layer_size_conv == 512 and dense_layer == 1 and layer_size == 64)\
                #     or (conv_layer == 2 and layer_size_conv == 128 and dense_layer == 1 and layer_size == 64)\
                #     or (conv_layer == 2 and layer_size_conv == 256 and dense_layer == 1 and layer_size == 64)\
                #     or (layer_size_conv == 512 and conv_layer == 2 and layer_size == 64 and dense_layer == 1)\
                #     or (layer_size_conv == 128 and conv_layer == 3 and layer_size == 128 and dense_layer == 1)\
                #     or (layer_size_conv == 256 and conv_layer == 3 and layer_size == 128 and dense_layer == 1)\
                #     or (layer_size_conv == 512 and conv_layer == 3 and layer_size == 128 and dense_layer == 1)\
                #     or (conv_layer == 1 and layer_size_conv == 128 and dense_layer == 1 and layer_size == 256)\
                #     or (conv_layer == 2 and layer_size_conv == 512 and dense_layer == 1 and layer_size == 128)\
                #     or (conv_layer == 3 and layer_size_conv == 512 and dense_layer == 2 and layer_size == 128)\
                #     or (conv_layer == 3 and layer_size_conv == 1024 and dense_layer == 2 and layer_size == 128):
                # #        or (conv_layer == 2 and layer_size_conv == 256and dense_layer == 1 and layer_size == 128):
                #     continue

                NAME = "THREE--{}-conv_nodes-{}-conv-{}-nodes-{}-dense-{}--CNN".format(layer_size_conv,conv_layer, layer_size, dense_layer, int(time.time()))
                print(conv_layer," ", layer_size_conv, "    ", dense_layer, " ", layer_size)
                log_dir = os.path.join(
                    "logs", NAME
                )
                tensorboard = TensorBoard(log_dir)

                model = Sequential()

                model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer-1):
                    model.add(Conv2D(layer_size_conv, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())
                for l in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))


                model.add(Dense(48))
                model.add(Activation('softmax'))

                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['acc'])
                model.fit(X, y, batch_size=64, epochs=2, validation_split=0.2, callbacks = [tensorboard])

model.save('2.0T-256-1-128-1.model')
