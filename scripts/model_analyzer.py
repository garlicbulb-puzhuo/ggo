from __future__ import print_function

import numpy as np
from keras.models import Model
from keras import backend as K
from data_proc import test_data_generator
from keras.models import load_model
from read_model_hdf5 import print_structure
import matplotlib.pyplot as plt
import theano.tensor as T


def get_layer_output(layer):
    def get_kth_layer_output(input):
        kth_layer_output = K.function([model.layers[0].input], [
                                      model.layers[layer].output])
        return kth_layer_output([input])[0]
    return get_kth_layer_output

# generate a test image
img_rows = 64
img_cols = 64
test_file = '../../../test_data.hdf5'
for imgs, masks, index in test_data_generator(test_file, img_rows, img_cols, iter=1):
    pass
im = np.array([imgs[600]])

# img: a test image; model_hdf5: train output


def plot_inner_layers_output(img, model, model_hdf5):
    layers = print_structure(model_hdf5)
    model.load_weights(model_hdf5)
    for m in range(len(layers) - 2):
        layer_kth_output = get_layer_output(m + 1)
        layer_output = layer_kth_output(im)
        num = layer_output.shape
        fig = plt.figure(m + 1)
        fig.suptitle(layers[m + 1] + ':' + str(num))
        k = 1
        for i in range(5):
            for j in range(5):
                n = '5' + '5' + str(k)
                plt.subplot(5, 5, k)
                plt.imshow(layer_output[0][k], cmap='Greys_r')
                k = k + 1
    plt.show()


def plot_input_for_layer(layer, model, img_rows, img_cols):
    layers = model.layers
    first_layer = model.layers[0]
    input_img = first_layer.input
    for m in layer:
        fig = plt.figure(m)
        fig.suptitle(layers[m])
        layer_output = layers[m].output
        for k in range(9):
            filter_index = k
            loss = T.mean(layer_output[:, filter_index, :, :])
            grads = T.grad(loss, input_img)[0]
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
            iterate = K.function(
                [input_img, K.learning_phase()], [loss, grads])
            input_img_data = np.random.random((1, 1, img_rows, img_cols)) * 20
            step = 0.05
            for i in range(100):
                loss_value, grads_value = iterate([input_img_data, 1])
                input_img_data += grads_value * step
            img = input_img_data[0][0]
            img = deprocess_image(img)
            plt.subplot(3, 3, k + 1)
            plt.imshow(img, cmap='Greys_r')
    plt.show()


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

if __name__ == "__main__":
    # path to the train output file
    from model_2 import get_unet
    model = get_unet()
    file_name = '../../../model2_newloss_train10_epoch50_20161126/unet.model2.model.iteration1.hdf5'
    img_rows = 128
    img_cols = 128
    plot_input_for_layer(
        layer=[1, 3, 10, 15], model=model, img_rows=img_rows, img_cols=img_cols)
