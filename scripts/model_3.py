'''Modification of VGG19 
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
'''
from __future__ import absolute_import
from __future__ import print_function

from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from optimizer import sgd
from loss import custom_loss, custom_metric


def get_model(input_shape=(1, 128, 128)):
    inputs = Input(input_shape)
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu',
                      border_mode='same', name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = Convolution2D(64, 3, 3, activation='relu',
                      border_mode='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu',
                      border_mode='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128, 3, 3, activation='relu',
                      border_mode='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu',
                      border_mode='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, activation='relu',
                      border_mode='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, activation='relu',
                      border_mode='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, activation='relu',
                      border_mode='same', name='block3_conv4')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu',
                      border_mode='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation='relu',
                      border_mode='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation='relu',
                      border_mode='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation='relu',
                      border_mode='same', name='block4_conv4')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu',
                      border_mode='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation='relu',
                      border_mode='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation='relu',
                      border_mode='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, 3, 3, activation='relu',
                      border_mode='same', name='block5_conv4')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same')(x)

    model = Model(input=inputs, output=x)

    model.compile(optimizer=sgd, loss=custom_loss, metrics=[custom_metric])
    return model, 'VGG19'
