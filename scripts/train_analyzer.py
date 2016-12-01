#!/usr/bin/env python2.7

import matplotlib.pyplot as plt
from csv import reader
import sys


def plot(data):
    # summarize history for loss and accuracy
    train_accuracy = [i[0] for i in data[::]]
    train_loss = [i[1] for i in data[::]]
    validation_accuracy = [i[2] for i in data[::]]
    validation_loss = [i[3] for i in data[::]]
    epoch = [i[5] for i in data[::]]

    # summarize history for accuracy
    plt.plot(train_accuracy)
    plt.plot(validation_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    # path to the train output file
    csv_file = sys.argv[1]
    with open(csv_file, 'r') as f:
        data = list(reader(f))

    plot(data)
