from __future__ import print_function

import cv2
import numpy as np
import random
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from data_proc import train_val_data_generator, test_data_generator

#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'


img_rows = 64
img_cols = 64

smooth = 100


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4],
                mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3],
                mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2],
                mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1],
                mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5),
                  loss=dice_coef_loss, metrics=[dice_coef])
    print(model.layers)
    return model


def train_and_predict():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint(
        'unet.hdf5', monitor='loss', save_best_only=True)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    nb_epoch = 2
    data_file = '../../../train_data.hdf5'
    for train_imgs, train_masks, train_index, val_imgs, val_masks, val_index in train_val_data_generator(data_file, train_batch_size=2, val_batch_size=1, img_rows =64, img_cols = 64, iter = 2):
        model.fit(train_imgs, train_masks, batch_size=32, nb_epoch=nb_epoch, validation_data=(val_imgs, val_masks), verbose=1, shuffle=True, callbacks=[model_checkpoint])

'''
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)

    model.load_weights('unet.hdf5')
    print(model.layers)
    test_file = '../../../test_data.hdf5'
    for imgs, masks, index in test_data_generator(test_file, img_rows, img_cols, iter=2):
        print('-' * 30)
        print('Predicting masks on test data...')
        print('-' * 30)
        mask_test = model.predict(imgs, verbose=1)
        print(mask_test)
'''
if __name__ == '__main__':
    train_and_predict()
