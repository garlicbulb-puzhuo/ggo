from __future__ import print_function

from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from optimizer import adam
from keras.layers.normalization import BatchNormalization
from loss import dice_coef_loss, dice_coef

BACKEND = K.backend()


def get_axis():
    return 1 if (BACKEND == 'theano') else 3

def BNConv(nb_filter, nb_row, nb_col, w_decay=None, subsample=(1, 1), border_mode="same"):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             border_mode=border_mode, activation="relu",
                             W_regularizer=l2(w_decay) if w_decay else None, init="he_normal")(input)
        return BatchNormalization(mode=0, axis=get_axis())(conv)
    return f

def get_model(input_shape=(1, 512, 512)):
    inputs = Input(input_shape)
    conv1 = BNConv(32, 3, 3)(inputs)
    conv1 = BNConv(32, 3, 3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BNConv(64, 3, 3)(pool1)
    conv2 = BNConv(64, 3, 3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BNConv(128, 3, 3)(pool2)
    conv3 = BNConv(128, 3, 3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BNConv(256, 3, 3)(pool3)
    conv4 = BNConv(256, 3, 3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = BNConv(512, 3, 3)(pool4)
    conv5 = BNConv(512, 3, 3)(conv5)
    conv5 = Dropout(0.5)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4],
                mode='concat', concat_axis=get_axis())
    conv6 = BNConv(256, 3, 3)(up6)
    conv6 = BNConv(256, 3, 3)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3],
                mode='concat', concat_axis=get_axis())
    conv7 = BNConv(128, 3, 3)(up7)
    conv7 = BNConv(128, 3, 3)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2],
                mode='concat', concat_axis=get_axis())
    conv8 = BNConv(64, 3, 3)(up8)
    conv8 = BNConv(64, 3, 3)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1],
                mode='concat', concat_axis=get_axis())
    conv9 = BNConv(32, 3, 3)(up9)
    conv9 = BNConv(32, 3, 3)(conv9)
    conv9 = Dropout(0.5)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])
    return model, 'UNET'
