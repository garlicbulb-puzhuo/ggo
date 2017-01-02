from __future__ import print_function

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from optimizer import adam
from keras.layers.normalization import BatchNormalization
from loss import custom_loss, custom_metric

def BNConv(nb_filter, nb_row, nb_col, w_decay = None, subsample=(1, 1), border_mode="same"):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                      border_mode=border_mode, activation="relu",
                      W_regularizer=l2(w_decay) if w_decay else None, init="he_normal")(input)
        return BatchNormalization(mode=0, axis=1)(conv)
    return f

def get_unet(input_shape=(1, 128, 128), lr=1e-5, dropout_prob=0.5):
    inputs = Input(input_shape)
    x = BNConv(8, 7, 7)(inputs)
    x = BNConv(16, 3, 3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = BNConv(32, 3, 3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = BNConv(64, 3, 3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = BNConv(64, 3, 3)(x)
    x = BNConv(64, 3, 3)(x)
    x = UpSampling2D(size=(2,2))(x)

    x = BNConv(64, 3, 3)(x)
    x = UpSampling2D(size=(2,2))(x)

    x = BNConv(32, 7, 7)(x)
    x = UpSampling2D(size=(2,2))(x)

    x = BNConv(16, 3, 3)(x)
    x = BNConv(8, 7, 7)(x)

    x = Convolution2D(1, 1, 1, activation='sigmoid')(x)
    
    model = Model(input=inputs, output=x)
    model.compile(optimizer=adam, loss=custom_loss, metrics=[custom_metric])
    return model, "model_1"
