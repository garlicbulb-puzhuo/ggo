from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from itertools import tee
import socket
from multiprocessing import Process
import six.moves.cPickle as pickle
from flask import Flask, request

try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

from elephas.utils.functional_utils import subtract_params

from elephas.spark_model import get_server_weights
from elephas.spark_model import put_deltas_to_server

from keras.models import model_from_yaml

from ..data_utils import train_val_generator


class CustomSparkWorker(object):
    '''
    Asynchronous Spark worker. This code will be executed on workers.
    '''

    def __init__(self, yaml, train_config, iteration, frequency, master_url, master_optimizer, master_loss, master_metrics, custom_objects, callbacks=[],
                 worker_callbacks=[], spark_worker_config=None):
        self.yaml = yaml
        self.train_config = train_config
        self.iteration = iteration
        self.frequency = frequency
        self.master_url = master_url
        self.master_optimizer = master_optimizer
        self.master_loss = master_loss
        self.master_metrics = master_metrics
        self.custom_objects = custom_objects
        self.callbacks = callbacks
        self.worker_callbacks = worker_callbacks
        self.spark_worker_config = spark_worker_config

    def train(self, data_iterator):
        '''
        Train a keras model on a worker and send asynchronous updates
        to parameter server
        '''
        files = list(data_iterator)

        model = model_from_yaml(self.yaml, self.custom_objects)
        model.compile(optimizer=self.master_optimizer,
                      loss=self.master_loss, metrics=self.master_metrics)

        nb_epoch = self.train_config['nb_epoch']
        batch_size = self.train_config.get('batch_size')

        img_rows = int(self.spark_worker_config.get('img_rows'))
        img_cols = int(self.spark_worker_config.get('img_cols'))
        train_batch_size = int(
            self.spark_worker_config.get('train_batch_size'))
        data_gen_iteration = int(
            self.spark_worker_config.get('data_gen_iteration'))
        samples_per_epoch = int(
            self.spark_worker_config.get('samples_per_epoch'))

        if self.frequency == 'epoch':
            data_generator = train_val_generator(file=files, batch_size=samples_per_epoch, train_size=train_batch_size, val_size=0, img_rows=img_rows,
                                                 img_cols=img_cols, iter=data_gen_iteration, train_or_val="train")

            for epoch in range(nb_epoch):
                weights_before_training = get_server_weights(self.master_url)
                model.set_weights(weights_before_training)
                self.train_config['nb_epoch'] = 1
                x_train, y_train = data_generator.next()

                if self.worker_callbacks:
                    for worker_callback in self.worker_callbacks:
                        worker_callback.on_epoch_start(
                            epoch=epoch, iteration=self.iteration, model=model)

                if x_train.shape[0] > batch_size:
                    history = model.fit(
                        x_train, y_train, callbacks=self.callbacks, **self.train_config)

                if self.worker_callbacks and history:
                    for worker_callback in self.worker_callbacks:
                        worker_callback.on_epoch_end(
                            epoch=epoch, iteration=self.iteration, model=model, history=history)

                weights_after_training = model.get_weights()
                deltas = subtract_params(
                    weights_before_training, weights_after_training)
                put_deltas_to_server(deltas, self.master_url)
        elif self.frequency == 'batch':
            raise Exception("epoch is the only supported frequency mode")
        else:
            print('Choose frequency to be either batch or epoch')
        yield []
