import os
import matplotlib.pyplot as plt


def plot_pred_mask(img, pred_mask, mask =None): 
    if mask is None: 
        f, plots = plt.subplots(2, 2, figsize=(10, 10))
        plots[0,0].imshow(img, cmap=plt.cm.bone) 
        plots[0,1].imshow(pred_mask, cmap=plt.cm.bone)
        plt.show()
    else: 
        f, plots = plt.subplots(2, 2, figsize=(10, 10))
        plots[0,0].imshow(img, cmap=plt.cm.bone) 
        plots[0,1].imshow(pred_mask, cmap=plt.cm.bone)
        plots[1,0].imshow(mask, cmap=plt.cm.bone)
        plt.show()


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]
