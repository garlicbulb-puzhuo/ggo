import numpy as np 
from glob import glob
from ..preprocess.watershed import load_scan, get_pixels_hu
import cv2

nodule_dir = '/Volumes/MacBook/kaggle3/all_watershed_mask_prediction/'
before_512 = '/Volumes/MacBook/kaggle3/all_watershed_processed_data/'
before_256 = '/Volumes/MacBook/kaggle3/all_watershed_processed_data_resize_256/'
before_128 = '/Volumes/MacBook/kaggle3/all_watershed_processed_data_resize_128/'
after_512 = '/Volumes/MacBook/kaggle3/all_watershed_processed_data_filter_by_nodule/'
after_256 = '/Volumes/MacBook/kaggle3/all_watershed_processed_data_resize_256_filter_by_nodule/'
after_128 = '/Volumes/MacBook/kaggle3/all_watershed_processed_data_resize_128_filter_by_nodule/'

before_raw_512 = '/Volumes/MacBook/kaggle3/stage1/'
after_raw_512 = '/Volumes/MacBook/kaggle3/raw_512/'
after_raw_256 = '/Volumes/MacBook/kaggle3/raw_256/'
after_raw_128 = '/Volumes/MacBook/kaggle3/raw_128/'

def resize(imgs,size):
	num = imgs.shape[0]
	resize_imgs = np.ndarray([num,1,size,size],dtype=np.float32)
	for i in range(imgs.shape[0]):
		resize_imgs[i,0,:,:] = cv2.resize(imgs[i][0], (size, size), interpolation=cv2.INTER_CUBIC)
	return resize_imgs

nodule_f = glob(nodule_dir+"*.npy")
for f in nodule_f: 
	fname = f.split("/")[-1].replace("mask_","")
	imgs = np.load(f)
	n = 0
	ix=[]
	i = 0 
	for img in imgs: 
		if np.sum(img[0])>100: 
			n += 1
			ix.append(i)
		i += 1
	print n
	if n==0: 
		ix = np.random.choice(imgs.shape[0])
	print "saving "+fname
	imgs = np.load(before_512+fname)
	np.save(after_512+fname,imgs[ix,:,:,:])
	imgs = np.load(before_256+fname)
	np.save(after_256+fname,imgs[ix,:,:,:])
	imgs = np.load(before_128+fname)
	np.save(after_128+fname,imgs[ix,:,:,:])

	scan = load_scan(before_raw_512+fname.replace(".npy",""))
	imgs = get_pixels_hu(scan)
	imgs = np.expand_dims(imgs,axis=1)
	np.save(after_raw_512,imgs[ix,:,:,:])
	resized = resize(imgs,256)
	np.save(after_raw_256,resized[ix,:,:,:])
	resized = resize(imgs,128)
	np.save(after_raw_128,resized[ix,:,:,:])
	
  

