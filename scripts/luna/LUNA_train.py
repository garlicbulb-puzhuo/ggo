from __future__ import print_function

import numpy as np
from glob import glob
from keras import backend as K

import os
import argparse
import ConfigParser
from random import shuffle
from ..train import get_standalone_model_callbacks


BACKEND = K.backend()
if BACKEND == 'theano':
    K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
else:
    K.set_image_dim_ordering('tf')


def get_input_shape(img_rows, img_cols):
    if BACKEND == 'theano':
        return (1, img_rows, img_cols)
    else:
        return (img_rows, img_cols, 1)


def get_data(imgs):
    if BACKEND == 'theano':
        return imgs
    else:
        data = np.rollaxis(imgs, 1, 4)
        return data


def data_generator(path, batch_size=2, img_rows=512, img_cols=512, shuffle_data=True):
    """
    A data generator that generates images and masks.

    :param path: train or val data directory in string.
    :param batch_size: batch size.
    :param img_rows: NOT USED
    :param img_cols: NOT USED
    :param shuffle_data: whether shuffling images. It's set to True for now.
    """
    input_path = os.path.join(path, "*Images_*.npy")
    f = glob(input_path)
    N = len(f)
    n = 0
    while True:
        if shuffle_data:
            shuffle(f)
        in_imgs = np.load(f[n])
        in_masks = np.load(f[n].replace("Images", "Masks"))
        num = in_imgs.shape[0]
        mean = np.mean(in_imgs)
        std = np.std(in_imgs)
        in_imgs -= mean  # images should already be standardized, but just in case
        in_imgs /= std
        if num > 0:
            if shuffle_data:
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


def get_latest_hdf5(model_save_directory):
    """
    Returns the latest hdf5 file.

    :return: latest hdf5 filename.
    """
    model_path = os.path.join(model_save_directory, "*.hdf5")
    files = filter(os.path.isfile, glob(model_path))
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    if len(files) > 0:
        return files[0]


def train_and_predict(train_path, val_path, train_config):
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    img_rows = int(train_config.get('img_rows'))
    img_cols = int(train_config.get('img_cols'))
    model_id = int(train_config.get('model_id'))

    use_existing_model = bool(train_config.get('use_existing_model', 'False'))

    nb_epoch = int(train_config.get('nb_epoch'))
    batch_size = int(train_config.get('batch_size'))
    validation_split = float(train_config.get('validation_split', 0.2))

    print('Config: use_existing_model: %s' % use_existing_model)
    print('Config: nb_epoch: %s' % nb_epoch)
    print('Config: batch_size: %s' % batch_size)

    if model_id == 1:
        from ..model_1 import get_model

    elif model_id == 2:
        from ..model_2 import get_model

    elif model_id == 3:
        from ..model_3 import get_model

    elif model_id == 4:
        from ..model_4 import get_model

    elif model_id == 5:
        from ..model_5 import get_model

    else:
        import sys
        print("model_id should be in [1..5].")
        sys.exit(1)

    input_shape = (1, img_rows, img_cols)
    model, model_name = get_model(input_shape)
    model_callbacks = get_standalone_model_callbacks(
        model_name=model_name, model_id=model_id, train_config=train_config)

    if use_existing_model:
        model_save_directory = train_config.get('model_save_directory', os.getcwd())
        latest_hdf5_file = get_latest_hdf5(model_save_directory)
        if latest_hdf5_file:
            print('Reading from existing model/weights file %s...' % latest_hdf5_file)
            model.load_weights(latest_hdf5_file)

    #
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970
    # I was able to run 20 epochs with a training set size of 320 and
    # batch size of 2 in about an hour. I started getting reseasonable masks
    # after about 3 hours of training.
    #
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    # TODO: we have to make this as an argument to determine whether we use
    # fit or fit_generator
    model.fit_generator(
        generator=data_generator(train_path, batch_size=batch_size),
        samples_per_epoch=2000, nb_epoch=nb_epoch, verbose=1, callbacks=model_callbacks, max_q_size=5, nb_worker=1,
        validation_data=data_generator(val_path, batch_size=batch_size),
        nb_val_samples=200)


def get_parser():
    parser = argparse.ArgumentParser(description='Train model.')

    parser.add_argument('-t', '--train-path', nargs='?', required=True,
                        help='train data directory containing numpy formatted images and masks in the training set')
    parser.add_argument('-v', '--val-path', nargs='?', required=True,
                        help='val data directory containing numpy formatted images and masks in the val set')
    parser.add_argument('-c', '--config-file', nargs='?', required=True,
                        help='config file for your training and prediction')
    return parser


def main(prog_args):
    parser = get_parser()
    args = parser.parse_args(prog_args)

    print(args)

    config = ConfigParser.ConfigParser()
    config.read(args.config_file)
    data_config = dict(config.items('config'))

    train_and_predict(train_path=args.train_path,
                      val_path=args.val_path, train_config=data_config)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
