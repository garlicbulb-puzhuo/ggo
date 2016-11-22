from __future__ import print_function

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from loss import dice_coef_loss

def custom_loss(y_true, y_pred): 
  return dice_coef_loss(y_true = y_true, y_pred = y_pred, weights =100)

def custom_metric(y_true, y_pred): 
  return -custom_loss(y_true, y_pred)

def get_unet(input_shape=(1, 128, 128), lr=1e-5, dropout_prob=0.5):
    inputs = Input(input_shape)
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(inputs)
    BN1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(BN1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    dropout1 = Dropout(dropout_prob)(pool1)

    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(dropout1)
    BN2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout2 = Dropout(dropout_prob)(pool2)

    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(dropout2)
    BN3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(BN3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    dropout3 = Dropout(dropout_prob)(pool3)

    conv4 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(dropout3)
    BN4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(BN4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    dropout4 = Dropout(dropout_prob)(pool4)

    conv5 = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same')(dropout4)
    BN5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same')(BN5)
    dropout5 = Dropout(dropout_prob)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(dropout5), conv4],
                mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(up6)
    BN6 = BatchNormalization()(conv6)
    conv6 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(BN6)
    dropout6 = Dropout(dropout_prob)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(dropout6), conv3],
                mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(up7)
    BN7 = BatchNormalization()(conv7)
    conv7 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(BN7)
    dropout7 = Dropout(dropout_prob)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2],
                mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    BN8 = BatchNormalization()(conv8)
    conv8 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(BN8)
    dropout8 = Dropout(dropout_prob)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1],
                mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    BN9 = BatchNormalization()(conv9)
    conv9 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(BN9)

    BN10 = BatchNormalization()(conv9)
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(BN10)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=lr),
                  loss=custom_loss, metrics=[custom_metric])
    return model
