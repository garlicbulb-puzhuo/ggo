#!/usr/bin/env python2.7

from __future__ import print_function

from keras import models
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback

import ConfigParser
import argparse
import sys
import logging
import numpy as np

from loss import custom_loss

from data_utils import train_val_data_generator, test_data_generator, train_val_generator

import signal
import traceback

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# match images and masks
logging_handler_out = logging.StreamHandler(sys.stdout)
logger.addHandler(logging_handler_out)


signal.signal(signal.SIGUSR1, lambda sig, stack: traceback.print_stack(stack))


def get_spark_model(model, model_name, model_id, train_config):
    from elephas.spark_model import SparkModel
    from elephas import optimizers as elephas_optimizers
    from pyspark import SparkContext, SparkConf
    from elephas.spark_model import ModelCallback
    from optimizer import adam

    master_server_port = int(train_config.get('master_server_port', 5000))
    worker_epoch_updates = int(
        train_config.get('worker_epoch_updates', 50))
    train_batch_size = int(train_config.get('train_batch_size'))

    logger.info("spark master server port : {0}".format(master_server_port))

    class SparkModelCheckPoint(ModelCallback):

        def __init__(self):
            self.worker_epoch_updates = worker_epoch_updates
            self.current_worker_epoch = 0

        def on_update_parameters(self, spark_model):
            self.current_worker_epoch += 1
            logger.info("get update parameters request from worker | aggregate epoch %d" %
                        self.current_worker_epoch)
            if self.current_worker_epoch % self.worker_epoch_updates == 0:
                # update spark model's internal master's weights
                spark_model.update_weights()

                # write parent spark model's weights
                logger.info(
                    'save intermediate model for model %s %d' % (model_name, model_id))
                models.save_model(spark_model.master_network,
                                  '%s.spark.model%d.acc_epoch%d.intermediate.hdf5' % (model_name, model_id, self.current_worker_epoch))

    spark_model_callback = SparkModelCheckPoint()

    conf = SparkConf().setAppName('Spark_Backend')
    sc = SparkContext(conf=conf)
    driver_adam = elephas_optimizers.Adam(clipnorm=1.0)

    spark_model = SparkModel(sc, model, optimizer=driver_adam, frequency='epoch',
                             mode='asynchronous', num_workers=4, master_optimizer=adam, master_loss=custom_loss, master_server_port=master_server_port,
                             model_callbacks=[spark_model_callback])

    return sc, spark_model


def get_standalone_model_callbacks(model_name, model_id, train_config):
    early_stop_min_delta = float(
        train_config.get('early_stop_min_delta', 1e-6))
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=early_stop_min_delta, patience=2, verbose=0)

    model_checkpoint = ModelCheckpoint(
        '%s.standalone.model%d.{epoch:02d}.hdf5' % (model_name, model_id), monitor='loss', save_best_only=False)

    standalone_loss_history_file = train_config.get(
        'standalone_loss_history_file', 'standalone_loss_history_file')

    class LossHistory(Callback):

        def __init__(self, filename):
            self.file = filename
            self.losses = []
            self.val_losses = []

        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []

        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            print("train_loss: {0}; val_loss: {1}".format(
                logs.get('loss'), logs.get('val_loss')))
            losses = np.vstack((self.losses, self.val_losses))
            np.savetxt(self.file, losses, delimiter=',')

    print_history = LossHistory(standalone_loss_history_file)

    class PrintBatch(Callback):

        def on_batch_end(self, epoch, logs={}):
            print(logs)

    pb = PrintBatch()
    return [model_checkpoint, print_history]


def get_spark_model_callbacks(model_name, model_id, train_config):
    from elephas.spark_model import SparkWorkerCallback
    import os
    import socket

    worker_epoch_updates = int(
        train_config.get('worker_epoch_updates', 50))

    class PrintHistoryCallback(Callback):

        def __init__(self):
            self.worker_epoch_updates = worker_epoch_updates
            self.current_worker_epoch = 0

        def on_epoch_end(self, epoch, logs={}):
            keys = logs.keys()
            values = logs.values()
            keys.append('hostname')
            keys.append('pid')
            keys.append('epoch')
            values.append(socket.gethostname())
            values.append(os.getpid())
            values.append(epoch)

            print()
            print("history and metadata keys: {0}".format(keys))
            print("history and metadata values: {0}".format(values))

    class SparkWorkerModelCheckpoint(SparkWorkerCallback):

        def __init__(self, model_filepath):
            self.model_filepath = model_filepath
            self.worker_epoch_updates = worker_epoch_updates
            self.losses = []
            self.val_losses = []

        def on_epoch_start(self, epoch, iteration, model):
            print()
            if epoch == 0:
                print(
                    "saving worker model at the start of each iteration: epoch %s" % epoch)
                model_filepath = self.model_filepath.format(
                    epoch=epoch, iteration=iteration)
                models.save_model(model, model_filepath)

        def on_epoch_end(self, epoch, iteration, model, history):
            print()
            epoch += 1
            if epoch % self.worker_epoch_updates == 0:
                print("saving worker model: epoch %s" % epoch)
                model_filepath = self.model_filepath.format(
                    epoch=epoch, iteration=iteration, **history.history)
                models.save_model(model, model_filepath)

            keys = history.history.keys()
            values = history.history.values()
            keys.append('hostname')
            keys.append('pid')
            keys.append('epoch')
            keys.append('iteration')
            values.append(socket.gethostname())
            values.append(os.getpid())
            values.append(epoch)
            values.append(iteration)

            print()
            print("history and metadata keys: {0}".format(keys))
            print("history and metadata values: {0}".format(values))

    early_stop_min_delta = float(
        train_config.get('early_stop_min_delta', 0.01))
    print_history = PrintHistoryCallback()
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=early_stop_min_delta, patience=2, verbose=0)
    spark_worker_callback = SparkWorkerModelCheckpoint(
        '%s.spark.model%d.{iteration:}.{epoch:02d}.hdf5' % (model_name, model_id))

    return [early_stop], [spark_worker_callback]


def train(train_imgs_path, train_mode, train_config):
    def transfer_existing_model():
        import glob
        import re
        import os

        files = glob.glob("*[model|weights]*.hdf5")
        li = 0

        if files:
            logger.info('Found existing model files in working directory')
            files.sort(key=os.path.getmtime)
            pattern = r".*iteration(\d+).*.hdf5"

            for model_file in files:
                logger.info('%s' % model_file)
                match = re.search(pattern, model_file)
                if match:
                    m = re.match(pattern, model_file)
                    li = int(m.group(0))
            print('-' * 30)
            print('Transferring from existing model...')
            print('-' * 30)
            logger.info(
                'Load weights from the last modified model file %s' % files[-1])
            model.load_weights(files[-1])
            return True, li
        else:
            logger.info('No existing model files in working directory')
            return False, li

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    img_rows = int(train_config.get('img_rows'))
    img_cols = int(train_config.get('img_cols'))

    model_id = int(train_config.get('model_id'))
    if model_id == 1:
        from model_1 import get_model

    if model_id == 2:
        from model_2 import get_model

    if model_id == 3:
        from model_3 import get_model

    if model_id == 4:
        import sys
        sys.setrecursionlimit(1000000)
        from model_4 import get_model

    input_shape = (1, img_rows, img_cols)
    model, model_name = get_model(input_shape)

    nb_epoch = int(train_config.get('nb_epoch'))
    train_batch_size = int(train_config.get('train_batch_size'))
    val_batch_size = int(train_config.get('val_batch_size'))
    data_gen_iteration = int(train_config.get('data_gen_iteration'))
    batch_size = int(train_config.get('batch_size'))

    samples_per_epoch = int(train_config.get('samples_per_epoch', 1000))
    nb_val_samples = int(train_config.get('nb_val_samples', 1000))

    if train_mode == 'spark':
        sc, spark_model = get_spark_model(
            model=model, model_name=model_name, model_id=model_id, train_config=train_config)
        model_callbacks, worker_callbacks = get_spark_model_callbacks(
            model_name=model_name, model_id=model_id, train_config=train_config)
    else:
        model_callbacks = get_standalone_model_callbacks(
            model_name=model_name, model_id=model_id, train_config=train_config)

    verbose = 1

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    if train_mode == 'standalone':
        from utils import listdir_fullpath
        import os
        if os.path.isdir(train_imgs_path):
            train_imgs_path = listdir_fullpath(train_imgs_path)

        model.fit_generator(
            generator=train_val_generator(file=train_imgs_path, batch_size=batch_size, train_size=train_batch_size, val_size=val_batch_size, img_rows=img_rows,
                                          img_cols=img_cols, iter=data_gen_iteration, train_or_val="train"),
            samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, verbose=verbose, callbacks=model_callbacks,
            validation_data=train_val_generator(file=train_imgs_path, batch_size=batch_size, train_size=train_batch_size,
                                                val_size=val_batch_size, img_rows=img_rows, img_cols=img_cols, iter=data_gen_iteration, train_or_val="val"),
            nb_val_samples=nb_val_samples)
    elif train_mode == 'spark':
        # transfer model weights
        transfer, last_iteration = transfer_existing_model()
        from utils import listdir_fullpath
        import os

        # get the list of hdf5 files
        if os.path.isdir(train_imgs_path):
            files = listdir_fullpath(train_imgs_path)
            rdd = sc.parallelize(files)
            rdd.repartition(rdd.getNumPartitions())

            # train data via spark
            print('-' * 30)
            print('Loading and preprocessing train data...')
            print('-' * 30)

            spark_model.train(rdd, iteration=0, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose,
                              validation_split=0.1, callbacks=model_callbacks, worker_callbacks=worker_callbacks,
                              spark_worker_class='spark_utils.CustomSparkWorker', spark_worker_config=train_config)
            models.save_model(
                model, '%s.spark.model%d.hdf5' % (model_name, model_id))
        else:
            iteration = 1
            for train_imgs, train_masks, train_index, val_imgs, val_masks, val_index in \
                    train_val_data_generator(file=train_imgs_path, train_batch_size=train_batch_size, val_batch_size=val_batch_size, img_rows=img_rows,
                                             img_cols=img_cols, iter=data_gen_iteration):
                print('-' * 30)
                print(
                    'Loading and preprocessing train data for iteration %d...' % iteration)
                print('-' * 30)

                logger.info(train_imgs.shape)

                from elephas.utils.rdd_utils import to_simple_rdd

                if transfer and iteration <= last_iteration:
                    logger.info('The current iteration is %d and less than or equal to the configured start iteration %d. Skip current iteration.' % (
                        iteration, last_iteration))
                    iteration += 1
                    continue

                rdd = to_simple_rdd(sc, train_imgs, train_masks)
                spark_model.train(rdd, iteration=iteration, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose,
                                  validation_split=0.1, callbacks=model_callbacks, worker_callbacks=worker_callbacks)

                models.save_model(
                    model, '%s.spark.model%d.batch%d.iteration%d.hdf5' % (model_name, model_id, train_batch_size, iteration))
                iteration += 1

    print('-' * 30)
    print('Training done')
    print('-' * 30)


def predict(model_file_path, test_imgs_path, config):
    img_rows = int(config.get('img_rows'))
    img_cols = int(config.get('img_cols'))

    model_id = int(config.get('model_id'))
    if model_id == 1:
        from model_1 import get_model

    if model_id == 2:
        from model_2 import get_model

    if model_id == 3:
        from model_3 import get_model

    model, model_name = get_model()
    model.load_weights(model_file_path)
    print(model.layers)

    for imgs, masks, index in test_data_generator(test_imgs_path, img_rows, img_cols, iter=2):
        print('-' * 30)
        print('Predicting masks on test data...')
        print('-' * 30)
        mask_test = model.predict(imgs, verbose=1)
        print(mask_test)


def get_parser():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--train', dest='train',
                        action='store_true', help='train the model')
    parser.add_argument('--train_imgs_path', metavar='train_imgs_path', nargs='?',
                        help='input HD5 file for images and masks in the training set')
    parser.add_argument('--train_mode', metavar='train_mode', nargs='?',
                        help='mode to train your model, can be either spark or standalone')
    parser.add_argument('--config_file', metavar='config_file', nargs='?',
                        help='config file for your training and prediction')
    parser.add_argument('--predict', dest='predict',
                        action='store_true', help='predict the model')
    parser.add_argument('--test_imgs_path', metavar='test_imgs_path', nargs='?',
                        help='input HD5 file for images and masks in the test set')
    parser.add_argument('--model_file_path', metavar='model_file_path', nargs='?',
                        help='the HD5 file to store the model and weights')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    if not args.train and not args.predict:
        parser.error('Required to set either --train or --predict option')

    if not args.config_file:
        parser.error('Required to set --config_file')

    if args.train and (args.train_imgs_path is None or args.train_mode is None):
        parser.error(
            'arguments --train_imgs_path and --train_mode are required when --train is specified')

    if args.predict and (args.test_imgs_path is None or args.model_file_path is None):
        parser.error(
            'arguments --test_imgs_path and --model_file_path are required when --predict is specified')

    config = ConfigParser.ConfigParser()
    config.read(args.config_file)
    data_config = dict(config.items('config'))

    if args.train:
        print("train images path %s" % args.train_imgs_path)
        train(train_imgs_path=args.train_imgs_path,
              train_mode=args.train_mode, train_config=data_config)

    if args.predict:
        predict(model_file_path=args.model_file_path,
                test_imgs_path=args.test_imgs_path,
                config=data_config)
