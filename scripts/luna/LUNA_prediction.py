from __future__ import print_function

import numpy as np
from glob import glob
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import os
import argparse
import ConfigParser
from ..loss import dice_coef_np
from ..utils import plot_pred_mask
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

def predict(model_id, model_weights, test_path, pred_name):
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    #img_rows = int(train_config.get('img_rows'))
    #img_cols = int(train_config.get('img_cols'))
    #model_id = int(train_config.get('model_id'))
    img_rows = 512
    img_cols = 512

    if model_id == 1:
        from ..model_1 import get_model

    if model_id == 2:
        from ..model_2 import get_model

    if model_id == 3:
        from ..model_3 import get_model

    if model_id == 4:
        import sys
        sys.setrecursionlimit(1000000)
        from ..model_4 import get_model

    input_shape = (1, img_rows, img_cols)
    model, model_name = get_model(input_shape)

    # loading model weights
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(model_weights)

    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    test_list = glob(test_path+"*Images*.npy")
    '''
    for f in test_list: 
        print("Processing: "+f)
        imgs = np.load(f)
        mean = np.mean(imgs)
        std = np.std(imgs)
        imgs-= mean  # images should already be standardized, but just in case
        imgs /= std
        num = len(imgs)
        mask_pred = np.ndarray([num,1,512,512],dtype=np.float32)
        mask_pred = get_layer_output([imgs,0])[0]
        np.save(f.replace('testImages', pred_name), mask_pred)
    '''
    mean = 0.0
    n = 1
    fn = 0 
    for f in test_list:
        mask = np.load(f.replace('testImages','testMasks'))
        mask_pred = np.load(f.replace('testImages',pred_name))
        img = np.load(f)
        num = len(mask)
        for i in range(num):
            mean+=dice_coef_np(mask[i,0], mask_pred[i,0])
            n +=1
            plot_pred_mask(img[i,0],mask_pred[i,0],mask[i,0])
            if np.all(mask_pred[i,0]<0.01):
                fn +=1

    mean/=n
    print("Mean Dice Coeff : ",mean)
    print(n)
    print(fn)



def get_parser():
    parser = argparse.ArgumentParser(description='Prediction.')
    #parser.add_argument('--working_path', metavar='working_path', nargs='?',
    #                    help='working directory containing numpy formatted images and masks in the training set')
    parser.add_argument('--test_path', metavar='train_path', nargs='?',
                        help='test data directory containing numpy formatted images and masks in the test set')
    #parser.add_argument('--config_file', metavar='config_file', nargs='?',
    #                    help='config file for your training and prediction')
    parser.add_argument('--model_id', metavar='model_id', nargs='?', type=int,
                         help='model_id')
    parser.add_argument('--model_weights', metavar='model_weights', nargs='?',
                         help='trained model weights file')
    parser.add_argument('--pred_name', metavar='pred_name', nargs='?',
                         help='predition mask name')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    if not args.model_id:
        parser.error('Required to set --model_id')

    if not args.test_path:
        parser.error('Required to set --test_path')

    predict(model_id=args.model_id, model_weights=args.model_weights, test_path=args.test_path, pred_name=args.pred_name)
