from keras import backend as K
import numpy as np
from theano import * 
import theano.tensor as T

def dice_coef_prob(y_true, y_pred, smooth, weights):
	if weights == None: 
		weights = 1
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	w_f = T.where(T.eq(y_true_f, 0), weights, 1)
	intersection = K.sum(y_true_f * y_pred_f * w_f * w_f) 
	return (2. * intersection + smooth) / (K.sum((y_true_f*w_f)**2) + K.sum((y_pred_f*w_f)**2) + smooth)

def dice_coef_binary(y_true, y_pred, smooth, weights, threshold=0.5):
	if weights == None: 
		weights = 1
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	y_true_f = T.where(T.le(y_true_f, threshold), 0, 1)
	y_pred_f = T.where(T.le(y_pred_f, threshold), 0, 1)
 	w_f = T.where(T.eq(y_true_f, 0), weights, 1)
	intersection = K.sum(T.eq(y_true_f,y_pred_f) * w_f )
	return (2. * intersection + smooth) / (K.sum(1 * w_f ) + K.sum(1 * w_f) + smooth)

def dice_coef_loss(y_true, y_pred, version = "prob", smooth= 1e-15, threshold=0.5, weights = None):
	if version == "prob": 
		return -dice_coef_prob(y_true, y_pred, smooth=smooth, weights = weights)
	if version == "binary": 
		return -dice_coef_binary(y_true, y_pred, smooth=smooth, weights = weights, threshold=threshold)

def custom_loss(y_true, y_pred): 
  return dice_coef_loss(y_true = y_true, y_pred = y_pred, weights =100)

def custom_metric(y_true, y_pred): 
  return -custom_loss(y_true, y_pred)