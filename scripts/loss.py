from keras import backend as K
import numpy as np
import theano.tensor as T


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
    return (2. * intersection +smooth) / (np.sum(weighted_y_true) + np.sum(weighted_y_pred) + smooth)
