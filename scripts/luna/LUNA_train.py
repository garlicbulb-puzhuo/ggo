from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as K

import os
import argparse
import ConfigParser

from ..train import get_standalone_model_callbacks


K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

smooth = 1.


# def get_standalone_model_callbacks(model_name, loss_history_file='loss_history_file'):
#     model_checkpoint = ModelCheckpoint(
#         '%s.standalone.model.{epoch:02d}.hdf5' % model_name, monitor='loss', save_best_only=False)
#
#     class LossHistory(Callback):
#
#         def __init__(self, filename):
#             self.file = filename
#             self.losses = []
#             self.val_losses = []
#
#         def on_train_begin(self, logs={}):
#             self.losses = []
#             self.val_losses = []
#
#         def on_epoch_end(self, epoch, logs={}):
#             self.losses.append(logs.get('loss'))
#             self.val_losses.append(logs.get('val_loss'))
#             print("train_loss: {0}; val_loss: {1}".format(
#                 logs.get('loss'), logs.get('val_loss')))
#             losses = np.vstack((self.losses, self.val_losses))
#             np.savetxt(self.file, losses, delimiter=',')
#
#     print_history = LossHistory(loss_history_file)
#
#     class PrintBatch(Callback):
#
#         def on_batch_end(self, epoch, logs={}):
#             print(logs)
#
#     pb = PrintBatch()
#     return [model_checkpoint, print_history]


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def train_and_predict(use_existing, working_path, train_config):
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    imgs_train = np.load(os.path.join(
        working_path, "trainImages.npy")).astype(np.float32)
    imgs_mask_train = np.load(os.path.join(
        working_path, "trainMasks.npy")).astype(np.float32)

    # imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
    # imgs_mask_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    img_rows = int(train_config.get('img_rows'))
    img_cols = int(train_config.get('img_cols'))
    model_id = int(train_config.get('model_id'))

    nb_epoch = int(train_config.get('nb_epoch'))
    batch_size = int(train_config.get('batch_size'))
    validation_split = float(train_config.get('validation_split', 0.2))

    if model_id == 1:
        from model_1 import get_model

    if model_id == 2:
        from model_2 import get_model

    if model_id == 3:
        from model_3 import get_model

    if model_id == 4:
        import sys
        sys.setrecursionlimit(1000000)
        from model_4 import get_model

    input_shape = (1, img_rows, img_cols)
    model, model_name = get_model(input_shape)
    model = get_model(input_shape)
    model_callbacks = get_standalone_model_callbacks(model_name=model_name)

    # Saving weights to unet.hdf5 at checkpoints
    # model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    #
    # Should we load existing weights?
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights('./unet.hdf5')

    #
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970
    # I was able to run 20 epochs with a training set size of 320 and
    # batch size of 2 in about an hour. I started getting reseasonable masks
    # after about 3 hours of training.
    #
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, shuffle=True, validation_split=validation_split, callbacks=model_callbacks)

    '''
    # loading best weights from training session
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('./unet.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
    np.save('masksTestPredicted.npy', imgs_mask_test)
    mean = 0.0
    for i in range(num_test):
        mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
    mean/=num_test
    print("Mean Dice Coeff : ",mean)
    '''


def get_parser():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--working_path', metavar='working_path', nargs='?',
                        help='working directory containing numpy formatted images and masks in the training set')
    parser.add_argument('--config_file', metavar='config_file', nargs='?',
                        help='config file for your training and prediction')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    if not args.config_file:
        parser.error('Required to set --config_file')

    if not args.working_path:
        parser.error('Required to set --working_path')

    config = ConfigParser.ConfigParser()
    config.read(args.config_file)
    data_config = dict(config.items('config'))

    train_and_predict(False, working_path=args.working_path,
                      train_config=data_config)
