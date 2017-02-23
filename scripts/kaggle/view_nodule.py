import matplotlib.pyplot as plt
from glob import glob 
import os
import pandas as pd
import numpy as np

#image_dir = '/Volumes/MacBook/kaggle3/watershed_processed_data/'
image_dir = '/Users/zhobuy98/workspace/kaggle3/output/kaggle3/watershed_processed_data/'
#nodule_dir = '/Volumes/MacBook/kaggle3/watershed_mask_prediction/'
nodule_dir = '/Users/zhobuy98/workspace/kaggle3/output/kaggle3//watershed_mask_prediction/'
label_path = '/Users/zhobuy98/workspace/kaggle3/input/stage1_labels.csv'

def get_cancer_id(nodule_dir, label_path): 
	flist = glob(os.path.join(nodule_dir+"*.npy"))
	pa = [q.replace("mask_","") for q in [p.replace(".npy","") for p in [f.replace(nodule_dir,"") for f in flist]]]
	df = pd.read_csv(label_path)
	cancer_id = []
	normal_id = []
	for i in pa: 
		if i in list(df['id']):
			if df.loc[df['id']==i,'cancer'].values[0]==1: 
				cancer_id.append(i)
			else:
				normal_id.append(i)
	return cancer_id, normal_id

def plot_nodule(image_dir,nodule_dir, cancer_id): 
	for p in cancer_id[1:]: 
		imgs = np.load(image_dir+p+".npy")
		nodules = np.load(nodule_dir+"mask_"+p+".npy")
		for i in range(nodules.shape[0]): 
			nodule = nodules[i,0,:,:]
			if np.any(nodule>0.9):
				f, plots = plt.subplots(1, 2, figsize=(10, 10))
				plots[0].imshow(imgs[i,0,:,:], cmap=plt.cm.bone) 
				plots[1].imshow(nodule, cmap=plt.cm.bone)
				plt.show()
	

if __name__ == "__main__":
	cancer_id, normal_id = get_cancer_id(nodule_dir,label_path)
	plot_nodule(image_dir,nodule_dir, normal_id)


