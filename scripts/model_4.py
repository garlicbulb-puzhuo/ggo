from keras.models import Model
from keras.layers import Input, Dense, Flatten, merge, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from optimizer import adam
from loss import custom_loss, custom_metric

# Evidently this model breaks Python's default recursion limit
# This is a theano issue
import sys
sys.setrecursionlimit(1000000)


def BNConv(nb_filter, nb_row, nb_col, w_decay, subsample=(1, 1), border_mode="same"):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                      border_mode=border_mode, activation="relu",
                      W_regularizer=l2(w_decay) if w_decay else None, init="he_normal")(input)
        return BatchNormalization(mode=0, axis=1)(conv)
    return f


def inception_v3(input_shape, dropout_prob, w_decay=None):
    inputs = Input(shape=input_shape)

    conv_1 = BNConv(32, 3, 3, w_decay, subsample=(1, 1), border_mode="same")(inputs)
    conv_2 = BNConv(32, 3, 3, w_decay, border_mode="same")(conv_1)
    conv_3 = BNConv(64, 3, 3, w_decay)(conv_2)

    pool_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="same")(conv_3)
    d = Dropout(dropout_prob)(pool_4)

    conv_5 = BNConv(80, 1, 1, w_decay)(d)
    conv_6 = BNConv(192, 3, 3, w_decay, border_mode="same")(conv_5)
    pool_7 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="same")(conv_6)
    d = Dropout(dropout_prob)(pool_7)

    inception_8 = InceptionFig5(w_decay)(d)
    inception_9 = InceptionFig5(w_decay)(inception_8)
    inception_10 = InceptionFig5(w_decay)(inception_9)

    inception_11 = DimReductionA(w_decay)(inception_10)

    inception_12 = InceptionFig6(w_decay)(inception_11)
    inception_13 = InceptionFig6(w_decay)(inception_12)
    inception_14 = InceptionFig6(w_decay)(inception_13)
    inception_15 = InceptionFig6(w_decay)(inception_14)
    inception_16 = InceptionFig6(w_decay)(inception_15)

    inception_17 = DimReductionB(w_decay)(inception_16)

    inception_18 = InceptionFig7(w_decay)(inception_17)
    inception_19 = InceptionFig7(w_decay)(inception_18)

    current_shape = (1280, input_shape[1]/16, input_shape[2]/16)

    x = Deconvolution2D(512, 3, 3, output_shape = (None, 512, input_shape[1]/8, input_shape[2]/8))(inception_19)
    x = Deconvolution2D(128, 3, 3, output_shape = (None, 1, input_shape[1]/4, input_shape[2]/4))(x)
    x = Deconvolution2D(32, 3, 3, output_shape = (None, 1, input_shape[1]/2, input_shape[2]/2))(x)
    x = Deconvolution2D(1, 3, 3, output_shape = (None, 1, input_shape[1], input_shape[2]))(x)
  
    model = Model(input = inputs, output = x)

    return model
   
    
def get_unet(input_shape = (1, 128, 128), dropout_prob = 0.5, w_decay = None):
    import sys
    sys.setrecursionlimit(1000000)
    import time
    start = time.time()
    model = inception_v3(input_shape, dropout_prob, w_decay)
    duration = time.time() - start
    print "{} s to make model".format(duration)
    start = time.time()
    model.output
    duration = time.time() - start
    print "{} s to get output".format(duration)
    start = time.time()
    model.compile(optimizer=adam, loss=custom_loss, metrics=[custom_metric])
    duration = time.time() - start
    print "{} s to get compile".format(duration)
    return model, 'inception_v3'
    

def InceptionFig5(w_decay):
    def f(input):

        # Tower A
        conv_a1 = BNConv(64, 1, 1, w_decay)(input)
        conv_a2 = BNConv(96, 3, 3, w_decay)(conv_a1)
        conv_a3 = BNConv(96, 3, 3, w_decay)(conv_a2)

        # Tower B
        conv_b1 = BNConv(48, 1, 1, w_decay)(input)
        conv_b2 = BNConv(64, 3, 3, w_decay)(conv_b1)

        # Tower C
        pool_c1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), border_mode="same")(input)
        conv_c2 = BNConv(64, 1, 1, w_decay)(pool_c1)

        # Tower D
        conv_d1 = BNConv(64, 1, 1, w_decay)(input)

        return merge([conv_a3, conv_b2, conv_c2, conv_d1], mode='concat', concat_axis=1)

    return f


def InceptionFig6(w_decay):
    def f(input):
        conv_a1 = BNConv(128, 1, 1, w_decay)(input)
        conv_a2 = BNConv(128, 1, 7, w_decay)(conv_a1)
        conv_a3 = BNConv(128, 7, 1, w_decay)(conv_a2)
        conv_a4 = BNConv(128, 1, 7, w_decay)(conv_a3)
        conv_a5 = BNConv(192, 7, 1, w_decay)(conv_a4)

        # Tower B
        conv_b1 = BNConv(128, 1, 1, w_decay)(input)
        conv_b2 = BNConv(128, 1, 7, w_decay)(conv_b1)
        conv_b3 = BNConv(192, 7, 1, w_decay)(conv_b2)

        # Tower C
        pool_c1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), border_mode="same")(input)
        conv_c2 = BNConv(192, 1, 1, w_decay)(pool_c1)

        # Tower D
        conv_d = BNConv(192, 1, 1, w_decay)(input)

        return merge([conv_a5, conv_b3, conv_c2, conv_d], mode="concat", concat_axis=1)

    return f


def InceptionFig7(w_decay):
    def f(input):
        # Tower A
        conv_a1 = BNConv(448, 1, 1, w_decay)(input)
        conv_a2 = BNConv(384, 3, 3, w_decay)(conv_a1)
        conv_a3 = BNConv(384, 1, 3, w_decay)(conv_a2)
        conv_a4 = BNConv(384, 3, 1, w_decay)(conv_a3)

        # Tower B
        conv_b1 = BNConv(384, 1, 1, w_decay)(input)
        conv_b2 = BNConv(384, 1, 3, w_decay)(conv_b1)
        conv_b3 = BNConv(384, 3, 1, w_decay)(conv_b2)

        # Tower C
        pool_c1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), border_mode="same")(input)
        conv_c2 = BNConv(192, 1, 1, w_decay)(pool_c1)

        # Tower D
        conv_d = BNConv(320, 1, 1, w_decay)(input)

        return merge([conv_a4, conv_b3, conv_c2, conv_d], mode="concat", concat_axis=1)

    return f


def DimReductionA(w_decay):
    def f(input):
        conv_a1 = BNConv(64, 1, 1, w_decay)(input)
        conv_a2 = BNConv(96, 3, 3, w_decay)(conv_a1)
        conv_a3 = BNConv(96, 3, 3, w_decay, subsample=(2, 2), border_mode="valid")(conv_a2)

        # another inconsistency between model.txt and the paper
        # the Fig 10 in the paper shows a 1x1 convolution before
        # the 3x3. Going with model.txt
        conv_b = BNConv(384, 3, 3, w_decay, subsample=(2, 2), border_mode="valid")(input)

        pool_c = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(input)

        return merge([conv_a3, conv_b, pool_c], mode="concat", concat_axis=1)
    return f


def DimReductionB(w_decay):
    def f(input):
        # Tower A
        conv_a1 = BNConv(192, 1, 1, w_decay)(input)
        conv_a2 = BNConv(320, 3, 3, w_decay, subsample=(2, 2), border_mode="valid")(conv_a1)

        # Tower B
        conv_b1 = BNConv(192, 1, 1, w_decay)(input)
        conv_b2 = BNConv(192, 1, 7, w_decay)(conv_b1)
        conv_b3 = BNConv(192, 7, 1, w_decay)(conv_b2)
        conv_b4 = BNConv(192, 3, 3, w_decay, subsample=(2, 2), border_mode="valid")(conv_b3)

        # Tower C
        pool_c = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(input)

        return merge([conv_a2, conv_b4, pool_c], mode="concat", concat_axis=1)
    return f


