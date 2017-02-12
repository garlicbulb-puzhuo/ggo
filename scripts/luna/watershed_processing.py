import sys
sys.path.insert(0, '../luna16_script/')

from watershed import load_scan, get_pixels_hu, seperate_lungs
import os
import fnmatch
import numpy as np

input_dir = '/Users/zhobuy98/workspace/kaggle3/input/stage1/'
output_path = '/Volumes/MacBook/kaggle3/watershed_processed_data/'
scans = [dirpath for dirpath, dirnames, files in os.walk(input_dir)][1:]

for scan in scans: 
	p = scan.replace(input_dir, "")
	print "processing " + p
	patient_scan = load_scan(scan)
	imgs = get_pixels_hu(patient_scan)
	num_images = imgs.shape[0]
	out_imgs = np.ndarray([num_images,1,512,512],dtype=np.float32)
	i = 0
	for img in imgs: 
		if i%10 == 0: 
			print 'imgage '+ str(i)
		segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed = seperate_lungs(img)
		out_imgs[i,0,:,:] = segmented
		i += 1
	np.save(output_path+p+".npy", out_imgs)
	




