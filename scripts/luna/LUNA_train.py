from __future__ import print_function

import numpy as np
from glob import glob
from keras import backend as K

import os
import argparse
import ConfigParser


if K.backend() == 'theano':
    K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
else:
    K.set_image_dim_ordering('tf')


def data_generator(path, batch_size=2, img_rows=512, img_cols=512, shuffle=True):
    """
    A data generator that generates images and masks.

    :param path: train or val data directory in string.
    :param batch_size: batch size.
    :param img_rows: NOT USED
    :param img_cols: NOT USED
    :param shuffle: whether shuffling images. It's set to True for now.
    """
    f = glob(path + "*Images_*.npy")
    N = len(f)
    n = 0
    while True:
        in_imgs = np.load(f[n])
        in_masks = np.load(f[n].replace("Images", "Masks"))
        num = in_imgs.shape[0]
        mean = np.mean(in_imgs)
        std = np.std(in_imgs)
        in_imgs -= mean  # images should already be standardized, but just in case
        in_imgs /= std
        if num > 0:
            if shuffle:
                ix = np.arange(num)
                np.random.shuffle(ix)
                in_imgs = in_imgs[ix, :, :, :]
                in_masks = in_masks[ix, :, :, :]
            if (num < batch_size):
                out = (in_imgs, in_masks)
                yield out
            else:
                k = 0
                while k + batch_size <= num:
                    out = (in_imgs[k:(k + batch_size), :, :, :],
                           in_masks[k:(k + batch_size), :, :, :])
                    k += batch_size
                    yield out
                if k < num:
                    out = (in_imgs[k:num, :, :, :], in_masks[k:num, :, :, :])
                    yield out
        n += 1
        if n >= N:
            n = 0


def get_latest_hdf5():
    """
    Returns the latest hdf5 file.

    :return: latest hdf5 filename.
    """
    files = filter(os.path.isfile, glob("*.hdf5"))
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    if len(files) > 0:
        return files[0]


def train_and_predict(use_existing, train_path, val_path, train_config):
    """
    Create/compile a model from model_[1,2,3,4] packages and fits the model on data generated batch-by-batch by a Python generator.

    :param use_existing: whether using existing model weights file. It's set to True for now.
    :param train_path: train data directory in string.
    :param val_path: val data directory in string.
    :param train_config: config in ConfigParser.ConfigParser().
    """
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
        from ..model_1 import get_model

    elif model_id == 2:
        from ..model_2 import get_model

    elif model_id == 3:
        from ..model_3 import get_model

    elif model_id == 4:
        from ..model_4 import get_model

    else:
        import sys
        print("model_id should be in [1..4].")
        sys.exit(1)

    from ..train import get_standalone_model_callbacks

    if K.image_dim_ordering() == 'th':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    model, model_name = get_model(input_shape)
    model_callbacks = get_standalone_model_callbacks(
        model_name=model_name, model_id=model_id, train_config=train_config)

    # Use existing latest model weights.
    if use_existing:
        latest_hdf5_file = get_latest_hdf5()
        if latest_hdf5_file:
            model.load_weights(latest_hdf5_file)

    #
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970
    # I was able to run 20 epochs with a training set size of 320 and
    # batch size of 2 in about an hour. I started getting reasonable masks
    # after about 3 hours of training.
    #
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    model.fit_generator(
        generator=data_generator(train_path, batch_size=batch_size),
        samples_per_epoch=2000, nb_epoch=nb_epoch, verbose=1, callbacks=model_callbacks, max_q_size=5, nb_worker=1,
        validation_data=data_generator(val_path, batch_size=batch_size),
        nb_val_samples=200)


def get_parser():
    parser = argparse.ArgumentParser(description='Train model.')

    parser.add_argument('--train_path', metavar='train_path', nargs='?', required=True,
                        help='train data directory containing numpy formatted images and masks in the training set')
    parser.add_argument('--val_path', metavar='val_path', nargs='?',
                        help='val data directory containing numpy formatted images and masks in the val set')
    parser.add_argument('--config_file', metavar='config_file', nargs='?', required=True,
                        help='config file for your training and prediction')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    config = ConfigParser.ConfigParser()
    config.read(args.config_file)
    data_config = dict(config.items('config'))

    train_and_predict(True, train_path=args.train_path,
                      val_path=args.val_path, train_config=data_config)
