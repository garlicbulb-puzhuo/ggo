#!/usr/bin/env python2.7

from __future__ import print_function

from keras import models
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import ConfigParser
import argparse
import sys
import logging

from data_utils import train_val_data_generator, test_data_generator


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# match images and masks
logging_handler_out = logging.StreamHandler(sys.stdout)
logger.addHandler(logging_handler_out)


def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet(img_rows, img_cols):
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


def get_spark_model(model):
    from elephas.spark_model import SparkModel
    from elephas import optimizers as elephas_optimizers
    from pyspark import SparkContext, SparkConf

    conf = SparkConf().setAppName('Spark_Backend')
    sc = SparkContext(conf=conf)
    adagrad = elephas_optimizers.Adagrad()
    spark_model = SparkModel(sc, model, optimizer=adagrad, frequency='epoch',
                             mode='asynchronous', num_workers=4, master_loss=dice_coef_loss)

    return sc, spark_model

'''
def train_val_split(imgs, masks, index, split_ratio = 0.1): 
    total = len(index)
    train = []
    val = []
    counter = 0 
    for i in index:
        r = random.random()
        if r < 0.1: 
            val.append(counter)
        else: 
            train.append(counter)
        counter += 1
    train_imgs = imgs[train,:,:,:]
    train_masks = masks[train,:,:,:]
    train_index = index[train]
    val_imgs = imgs[val,:,:,:]
    val_masks = masks[val,:,:,:]
    val_index = index[val]
    return train_imgs, train_masks, train_index, val_imgs, val_masks, val_index
'''


def train(train_imgs_path, train_mode, train_config):
    from elephas.utils.rdd_utils import to_simple_rdd

    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    img_rows = int(train_config.get('img_rows'))
    img_cols = int(train_config.get('img_cols'))

    ### import model
    from model_2 import get_unet_BN
    input_shape = (1, img_rows, img_cols)

    train_output = 'unet123_BN.hdf5'
    model = get_unet_BN(input_shape)

    if train_mode == 'spark':
        sc, spark_model = get_spark_model(model)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    nb_epoch = train_config('nb_epoch')
    train_batch_size = train_config('train_batch_size')
    val_batch_size = train_config('val_batch_size')
    verbose = 1
    iteration = 1

    for train_imgs, train_masks, train_index, val_imgs, val_masks, val_index in \
            train_val_data_generator(train_imgs_path, train_batch_size=train_batch_size, val_batch_size=val_batch_size, img_rows=img_rows, img_cols=img_cols):
        print(train_imgs.shape)

        if train_mode == 'spark':
            from elephas.spark_model import HistoryCallback

            class GgoHistoryCallback(HistoryCallback):

                def __init__(self):
                    pass

                def on_receive_history(self, history, metadata):
                    # list all data in history
                    print("history and metadata keys: {0}, {1}".format(
                        history.history.keys(), metadata.keys()))
                    print("history and metadata values: {0}, {1}".format(
                        history.history.values(), metadata.values()))

            history_callback = GgoHistoryCallback()
            rdd = to_simple_rdd(sc, train_imgs, train_masks)
            spark_model.train(rdd, batch_size=32, nb_epoch=nb_epoch, verbose=verbose,
                              validation_split=0.1, history_callback=history_callback)
            model.save('unet.model1.%d.hdf5' % iteration)
            models.save_model(model, 'unet.model2.%d.hdf5' % iteration)
            model.save_weights('unet.weights.%d.hdf5' % iteration)
        else:
            model_checkpoint = ModelCheckpoint(
                'unet.hdf5', monitor='loss', save_best_only=True)
            model.fit(train_imgs, train_masks, batch_size=32, nb_epoch=nb_epoch, validation_data=(val_imgs, val_masks), verbose=verbose, shuffle=True,
                      callbacks=[model_checkpoint])

        iteration += 1

    '''
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    '''

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)


def predict(model_file_path, test_imgs_path, config):
    img_rows = int(config.get('img_rows'))
    img_cols = int(config.get('img_cols'))
    model = get_unet(img_rows=img_rows, img_cols=img_cols)
    model.load_weights(model_file_path)
    print(model.layers)

    for imgs, masks, index in test_data_generator(test_imgs_path, img_rows, img_cols, iter=2):
        print('-' * 30)
        print('Predicting masks on test data...')
        print('-' * 30)
        mask_test = model.predict(imgs, verbose=1)
        print(mask_test)


def get_parser():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--train', dest='train',
                        action='store_true', help='train the model')
    parser.add_argument('--train_imgs_path', metavar='train_imgs_path', nargs='?',
                        help='input HD5 file for images and masks in the training set')
    parser.add_argument('--train_mode', metavar='train_mode', nargs='?',
                        help='mode to train your model, can be either spark or standalone')
    parser.add_argument('--config_file', metavar='config_file', nargs='?',
                        help='config file for your training and prediction')
    parser.add_argument('--predict', dest='predict',
                        action='store_true', help='predict the model')
    parser.add_argument('--test_imgs_path', metavar='test_imgs_path', nargs='?',
                        help='input HD5 file for images and masks in the test set')
    parser.add_argument('--model_file_path', metavar='model_file_path', nargs='?',
                        help='the HD5 file to store the model and weights')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    if not args.train and not args.predict:
        parser.error('Required to set either --train or --predict option')

    if not args.config_file:
        parser.error('Required to set --config_file')

    if args.train and (args.train_imgs_path is None or args.train_mode is None):
        parser.error(
            'arguments --train_imgs_path and --train_mode are required when --train is specified')

    if args.predict and (args.test_imgs_path is None or args.model_file_path is None):
        parser.error(
            'arguments --test_imgs_path and --model_file_path are required when --predict is specified')

    config = ConfigParser.ConfigParser()
    config.read(args.config_file)
    data_config = dict(config.items('config'))

    if args.train:
        print("train images path %s" % args.train_imgs_path)
        train(train_imgs_path=args.train_imgs_path,
              train_mode=args.train_mode, train_config=data_config)

    if args.predict:
        predict(model_file_path=args.model_file_path,
                test_imgs_path=args.test_imgs_path,
                config=data_config)
