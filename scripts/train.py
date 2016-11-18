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

### image rows and cols
img_rows = 64
img_cols = 64

def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def train_model():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    ### import model 
    from model_2 import get_unet_BN
    input_shape = (1, 64, 64)
    model = get_unet_BN(input_shape)
    train_output = 'unet123_BN.hdf5'
    model_checkpoint = ModelCheckpoint(
        train_output, monitor='loss', save_best_only=True)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    nb_epoch = 100
    data_file = '../../../train_data_123.hdf5'
    for train_imgs, train_masks, train_index, val_imgs, val_masks, val_index in train_val_data_generator(data_file, train_batch_size=2, val_batch_size=1, img_rows =img_rows, img_cols = img_cols, iter = 2):
        model.fit(train_imgs, train_masks, batch_size=32, nb_epoch=nb_epoch, validation_data=(val_imgs, val_masks), verbose=1, shuffle=True, callbacks=[model_checkpoint])


if __name__ == "__main__": 
    train_model()
