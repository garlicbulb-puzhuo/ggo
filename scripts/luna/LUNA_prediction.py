from __future__ import print_function

import numpy as np
from glob import glob
from keras import backend as K

import argparse
from ..loss import dice_coef_np
from ..utils import plot_pred_mask

# TODO: we will use tensorflow later
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code


def predict(model_id, model_weights, test_path, pred_name):
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
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(model_weights)

    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    test_list = glob(test_path+"*Images*.npy")
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
    parser.add_argument('-t', '--test-path', metavar='train_path', nargs='?', required=True,
                        help='test data directory containing numpy formatted images and masks in the test set')
    parser.add_argument('-m', '--model-id', metavar='model_id', nargs='?', type=int, required=True,
                        help='model_id')
    parser.add_argument('-w', '--model-weights', metavar='model_weights', nargs='?',
                        help='trained model weights file')
    parser.add_argument('-p', '--pred-name', metavar='pred_name', nargs='?',
                        help='prediction mask name')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    predict(model_id=args.model_id, model_weights=args.model_weights, test_path=args.test_path, pred_name=args.pred_name)
