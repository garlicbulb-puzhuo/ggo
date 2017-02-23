from keras import backend as K
import numpy as np
from theano import *
import theano.tensor as T

'''
def dice_coef_prob(y_true, y_pred, smooth, weights):
    if weights == None:
        weights = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    w_f = T.where(T.eq(y_true_f, 0), weights, 1)
    intersection = K.sum(y_true_f * y_pred_f * w_f * w_f)
    return (2. * intersection + smooth) / (K.sum((y_true_f * w_f)**2) + K.sum((y_pred_f * w_f)**2) + smooth)


def dice_coef_binary(y_true, y_pred, smooth, weights, threshold=0.5):
    if weights == None:
        weights = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = T.where(T.le(y_true_f, threshold), 0, 1)
    y_pred_f = T.where(T.le(y_pred_f, threshold), 0, 1)
    w_f = T.where(T.eq(y_true_f, 0), weights, 1)
    intersection = K.sum(T.eq(y_true_f, y_pred_f) * w_f)
    return (2. * intersection + smooth) / (K.sum(1 * w_f) + K.sum(1 * w_f) + smooth)


def dice_coef_loss(y_true, y_pred, version="prob", smooth=1e-5, threshold=0.5, weights=None):
    if version == "prob":
        return -dice_coef_prob(y_true, y_pred, smooth=smooth, weights=weights)
    if version == "binary":
        return -dice_coef_binary(y_true, y_pred, smooth=smooth, weights=weights, threshold=threshold)

def custom_loss(y_true, y_pred):
    return dice_coef_loss(y_true=y_true, y_pred=y_pred, weights=100)


def custom_metric(y_true, y_pred):
    return -custom_loss(y_true, y_pred)
'''

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection+smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def weighted_dice_coef(y_true, y_pred, weight=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    w_f = T.where(T.eq(y_true_f, 0), 1, weight)
    weighted_y_true = (y_true_f*w_f)/K.sum(w_f)
    weighted_y_pred = (y_pred_f*w_f)/K.sum(w_f)
    intersection = K.sum(weighted_y_true * weighted_y_pred)
    return (2. * intersection +smooth) / (K.sum(weighted_y_true) + K.sum(weighted_y_pred) + smooth)

def weighted_dice_coef_loss(y_true, y_pred, weight=100):
    return -weighted_dice_coef(y_true, y_pred, weight=weight)

def weighted_dice_coef_np(y_true, y_pred, weight=100):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    w_f = np.ones_like(y_true_f)
    w_f[np.where(y_true_f==1)] = weight
    weighted_y_true = (y_true_f*w_f)/np.sum(w_f)
    weighted_y_pred = (y_pred_f*w_f)/np.sum(w_f)
    intersection = np.sum(weighted_y_true * weighted_y_pred)
    #print intersection, np.sum(weighted_y_true), np.sum(weighted_y_pred)
    return (2. * intersection +smooth) / (np.sum(weighted_y_true) + np.sum(weighted_y_pred) + smooth)
