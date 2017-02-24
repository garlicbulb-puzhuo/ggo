from __future__ import print_function

import argparse
from glob import glob
import os
import numpy as np
from keras import backend as K


def predict(model_id, model_weights, input_path, output_path):
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

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
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights(model_weights)

    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [
                                  model.layers[-1].output])

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)

    input_files_path = os.path.join(input_path, "*.npy")
    f_list = glob(input_files_path)
    print("Total patients:" + str(len(f_list)))

    for f in f_list:
        print("Processing: " + f)
        imgs = np.load(f)
        mean = np.mean(imgs)
        std = np.std(imgs)
        imgs -= mean
        imgs /= std
        num = len(imgs)
        mask_pred = np.ndarray([num, 1, 512, 512], dtype=np.float32)
        batch_size = 50
        if (batch_size > num):
            mask_pred = get_layer_output([imgs, 0])[0]
        else:
            nbatch = num / batch_size
            remain = num % batch_size
            for i in range(nbatch):
                print("Batch:" + str(i))
                start_index = i * batch_size
                end_index = (i + 1) * batch_size
                mask_pred[start_index:end_index, :, :, :] = get_layer_output(
                    [imgs[start_index:end_index, :, :, :], 0])[0]
            if remain > 0:
                mask_pred[end_index:num, :, :, :] = get_layer_output(
                    [imgs[end_index:num, :, :, :], 0])[0]
        n = 0
        for i in range(num):
            if np.any(mask_pred[i, 0] > 0.9):
                n += 1
        print("positive images: " + str(n))
        output_file_path = os.path.join(output_path, 'mask_' + f.split("/")[-1])
        np.save(output_file_path, mask_pred)


def get_parser():
    parser = argparse.ArgumentParser(description='Prediction.')
    parser.add_argument('-i', '--input-path', nargs='?', required=True,
                        help='test data directory containing numpy formatted images and masks in the test set')
    parser.add_argument('-m', '--model-id', nargs='?', type=int, required=True,
                        help='model_id')
    parser.add_argument('-w', '--model-weights', nargs='?', required=True,
                        help='trained model weights file')
    parser.add_argument('-o', '--output-path', nargs='?', required=True,
                        help='output directory to store predition ')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    predict(model_id=args.model_id, model_weights=args.model_weights,
            input_path=args.input_path, output_path=args.output_path)
