import numpy as np
np.random.seed(1712)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import statistics
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from scipy.misc import imread, imresize



def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(60, 60, 3), name = 'zpd'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.load_weights('../input/vggweights/vgg16_weights.h5', by_name = True)

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

img_width, img_height = 60, 60
batch_size = 32
epochs = 10
nfolds = 10

# ----------------------------------------- Reading Driver Data --------------------------------------------

path = os.path.join('..', 'input/state-farm-distracted-driver-detection', 'driver_imgs_list.csv')
driver_imgs_list = pd.read_csv(path)
driver_imgs_list.head()
driverInImage = {}
drivers_list = []
driver_imgs = {}
classOfImage = {}
for index, row in driver_imgs_list.iterrows():
    driverInImage[row['img'].split(".")[0]] = row['subject']
    classOfImage[row['img'].split(".")[0]] = row['classname']
    if not row['subject'] in  driver_imgs:
        driver_imgs[row['subject']] = []
    driver_imgs[row['subject']].append(row['img'].split(".")[0])
    if not row['subject'] in drivers_list:
        drivers_list.append(row['subject'])

for train_drivers, validation_drivers in KFold(n_splits=nfolds, shuffle=False, random_state=None).split(drivers_list):
    validation_X = []
    validation_y = []
    train_X = []
    train_y = []
    for i in range(len(train_drivers)):
        for j in range(len(driver_imgs[drivers_list[train_drivers[i]]])):
            path = os.path.join('..', 'input/state-farm-distracted-driver-detection', 'train', classOfImage[driver_imgs[drivers_list[train_drivers[i]]][j]], driver_imgs[drivers_list[train_drivers[i]]][j] + '.jpg')
            resized_img = cv2.resize(cv2.imread(path), (img_width, img_height))
            train_X.append(resized_img)
            train_y.append(classOfImage[driver_imgs[drivers_list[train_drivers[i]]][j]][1])
    
    for i in range(len(validation_drivers)):
        for j in range(len(driver_imgs[drivers_list[validation_drivers[i]]])):
            path = os.path.join('..', 'input/state-farm-distracted-driver-detection', 'train', classOfImage[driver_imgs[drivers_list[validation_drivers[i]]][j]], driver_imgs[drivers_list[validation_drivers[i]]][j] + '.jpg')
            resized_img = cv2.resize(cv2.imread(path), (img_width, img_height))
            validation_X.append(resized_img)
            validation_y.append(classOfImage[driver_imgs[drivers_list[validation_drivers[i]]][j]][1])
    
    
    train_X = np.array(train_X, dtype=np.uint8)
    train_y = np.array(train_y, dtype=np.uint8)
    
    validation_X = np.array(validation_X, dtype=np.uint8)
    validation_y = np.array(validation_y, dtype=np.uint8)

    print('Reshape...')
    train_X = train_X.transpose((0, 1, 2, 3))
    validation_X = validation_X.transpose((0, 1, 2, 3))
    
    train_X = train_X.astype('float16')
    validation_X = validation_X.astype('float16')
    
    mean_pixel = [103.939, 116.779, 123.68]
    train_X[:, 0, :, :] -= mean_pixel[0]
    train_X[:, 1, :, :] -= mean_pixel[1]
    train_X[:, 2, :, :] -= mean_pixel[2]
    
    validation_X[:, 0, :, :] -= mean_pixel[0]
    validation_X[:, 1, :, :] -= mean_pixel[1]
    validation_X[:, 2, :, :] -= mean_pixel[2]
    
    train_y = np_utils.to_categorical(train_y, 10)
    validation_y = np_utils.to_categorical(validation_y, 10)
                
    print('Train shape:', train_X.shape)
    print(train_X.shape[0], 'train samples')
    
    model = VGG_16()
    model.fit(train_X, train_y, batch_size=batch_size, nb_epoch=epochs, shuffle=True, verbose=1, validation_data=(validation_X, validation_y))
    
    predictions_valid = model.predict(validation_X.astype('float32'), batch_size=batch_size, verbose=1)
    score = log_loss(validation_y, predictions_valid)
    print('Score log_loss: ', score)
    
    
        

'''for k in range(len(drivers_list)):
    validation_X = []
    validation_y = []
    train_X = []
    train_y = []
    for i in range(len(drivers_list)):
        for j in range(len(driver_imgs[drivers_list[i]])):
            path = os.path.join('..', 'input/state-farm-distracted-driver-detection', 'train', classOfImage[driver_imgs[drivers_list[i]][j]], driver_imgs[drivers_list[i]][j] + '.jpg')
            resized_img = cv2.resize(cv2.imread(path), (img_width, img_height))
            if i == k:
                validation_X.append(resized_img)
                validation_y.append(classOfImage[driver_imgs[drivers_list[i]][j]][1])
            else:
                train_X.append(resized_img)
                train_y.append(classOfImage[driver_imgs[drivers_list[i]][j]][1])
    
    train_X = np.array(train_X, dtype=np.uint8)
    train_y = np.array(train_y, dtype=np.uint8)

    print('Reshape...')
    train_X = train_X.transpose((0, 1, 2, 3))
    print('Train Data Shape :',train_X.shape)
    print('Convert to float...')
    train_X = train_X.astype('float16')
    mean_pixel = [103.939, 116.779, 123.68]
    print('Substract 0...')
    train_X[:, 0, :, :] -= mean_pixel[0]
    print('Substract 1...')
    train_X[:, 1, :, :] -= mean_pixel[1]
    print('Substract 2...')
    train_X[:, 2, :, :] -= mean_pixel[2]

    train_y = np_utils.to_categorical(train_y, 10)
                
    print('Train shape:', train_X.shape)
    print(train_X.shape[0], 'train samples')'''


