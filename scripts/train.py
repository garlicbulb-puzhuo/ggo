#!/usr/bin/env python2.7

from __future__ import print_function

from keras import models
from keras.callbacks import ModelCheckpoint

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


def get_spark_model(model, master_server_port):
    from elephas.spark_model import SparkModel
    from elephas import optimizers as elephas_optimizers
    from pyspark import SparkContext, SparkConf

    conf = SparkConf().setAppName('Spark_Backend')
    sc = SparkContext(conf=conf)
    adagrad = elephas_optimizers.Adagrad()
    spark_model = SparkModel(sc, model, optimizer=adagrad, frequency='epoch',
                             mode='asynchronous', num_workers=4, master_loss=dice_coef_loss, master_server_port=master_server_port)

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
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    img_rows = int(train_config.get('img_rows'))
    img_cols = int(train_config.get('img_cols'))

    model_id = int(train_config.get('model_id'))
    if model_id == 1:
        from model_1 import get_unet

    if model_id == 2:
        from model_2 import get_unet

    if model_id == 3:
        from model_3 import get_unet

    ### import model
    input_shape = (1, img_rows, img_cols)

    # train_output = 'unet123_BN.hdf5'

    model = get_unet(input_shape)

    if train_mode == 'spark':
        master_server_port = int(train_config.get('master_server_port', 5000))
        print("spark master server port : {0}".format(master_server_port))
        sc, spark_model = get_spark_model(model=model, master_server_port=master_server_port)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    nb_epoch = int(train_config.get('nb_epoch'))
    train_batch_size = int(train_config.get('train_batch_size'))
    val_batch_size = int(train_config.get('val_batch_size'))
    data_gen_iteration = int(train_config.get('data_gen_iteration'))
    verbose = 1
    iteration = 1

    for train_imgs, train_masks, train_index, val_imgs, val_masks, val_index in \
            train_val_data_generator(file=train_imgs_path, train_batch_size=train_batch_size, val_batch_size=val_batch_size, img_rows=img_rows,
                                     img_cols=img_cols, iter=data_gen_iteration):
        print(train_imgs.shape)

        if train_mode == 'spark':
            from elephas.utils.rdd_utils import to_simple_rdd
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

    model_id = int(config.get('model_id'))
    if model_id == 1:
        from model_1 import get_unet

    if model_id == 2:
        from model_2 import get_unet

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
