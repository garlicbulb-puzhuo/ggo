from __future__ import print_function

import numpy as np
from keras.models import Model
from keras import backend as K
from data_proc import test_data_generator
from train import get_unet
from keras.models import load_model
from read_model_hdf5 import print_structure
import matplotlib.pyplot as plt


def get_layer_output(layer): 
	def get_kth_layer_output(input):
		kth_layer_output = K.function([model.layers[0].input], [model.layers[layer].output])
		return kth_layer_output([input])[0]
	return get_kth_layer_output

#generate a test image 
img_rows = 64
img_cols = 64 
test_file = '../../../test_data.hdf5'
for imgs, masks, index in test_data_generator(test_file, img_rows, img_cols, iter=1): 
	pass
im = np.array([imgs[600]])

#img: a test image; model_hdf5: train output
def plot_inner_layers(img, model, model_hdf5):
	layers = print_structure(model_hdf5)
	model.load_weights(model_hdf5)
	for m in range(len(layers)-2): 
		layer_kth_output = get_layer_output(m+1)
		layer_output = layer_kth_output(im)
		num = layer_output.shape
		print(num)
		fig = plt.figure(m+1)
		fig.suptitle(layers[m+1]+':'+str(num))
		k = 1
		for i in range(5):
		    for j in range(5):
		        n = '5'+'5'+str(k)
		        plt.subplot(5,5,k)
		        plt.imshow(layer_output[0][k])
		        k = k + 1
	plt.show()



'''
weights = model.trainable_weights # weight tensors
weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors

input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]
get_gradients = K.function(inputs=input_tensors, outputs=gradients)
get_gradients(inputs)
'''
if __name__ == "__main__":
	#path to the train output file 
	file_name = 'unet.hdf5'
	model = get_unet()
	plot_inner_layers(img=im, model=model, model_hdf5=file_name)

