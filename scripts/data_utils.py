#!/usr/bin/env python2.7

import h5py
import numpy as np

patient_group_dict = {}


def list_all_patients(name, obj):
    if 'indices' in name:
        p_set = set(obj[:, 0])
        for p in p_set:
            patient_group_dict[p] = name[:-7]


def load_data_from_hdf5(file, patientID, patient_group_dict):
    f = h5py.File(file, 'r')
    group = patient_group_dict[patientID]
    imgs = f.get(group).get('imgs')
    masks = f.get(group).get('masks')
    indices = f.get(group).get('indices')
    ix = indices[:, 0] == patientID
    return imgs[ix, :, :, :], masks[ix, :, :, :], indices[ix, :]


# train_batch_size: number of train patients; val_batch_size: number of
# validation patients; iter: number of (train, val) generator
def train_val_data_generator(file, img_rows, img_cols, train_batch_size=1, val_batch_size=1, iter=1000, train_or_val = "both"):
    f = h5py.File(file, 'r')
    f.visititems(list_all_patients)
    p_list = patient_group_dict.keys()
    remaining = len(p_list)
    train_counter = 0
    val_counter = train_counter + train_batch_size
    if train_batch_size + val_batch_size > remaining:
        print 'Not enough data!'
    counter = 0
    while train_batch_size + val_batch_size < remaining and counter < iter:
        p_sublist = p_list[train_counter:(train_counter + train_batch_size)]
        train_imgs = np.array([]).reshape((0, 1, img_rows, img_cols))
        train_masks = np.array([]).reshape((0, 1, img_rows, img_cols))
        train_index = np.array([]).reshape((0, 4))
        for p in p_sublist:
            imgs, masks, indices = load_data_from_hdf5(
                file, p, patient_group_dict)
            train_imgs = np.vstack((train_imgs, imgs))
            train_masks = np.vstack((train_masks, masks))
            train_index = np.vstack((train_index, indices))
        val_imgs = np.array([]).reshape((0, 1, img_rows, img_cols))
        val_masks = np.array([]).reshape((0, 1, img_rows, img_cols))
        val_index = np.array([]).reshape((0, 4))
        if val_batch_size > 0:
            p_sublist = p_list[val_counter:(val_counter + val_batch_size)]
            imgs_len = 0
            for p in p_sublist:
                imgs, masks, indices = load_data_from_hdf5(
                    file, p, patient_group_dict)
                imgs_len += imgs.shape[0]
                val_imgs = np.vstack((val_imgs, imgs))
                val_masks = np.vstack((val_masks, masks))
                val_index = np.vstack((val_index, indices))
            val_counter = train_counter + train_batch_size
        train_counter = train_counter + train_batch_size + val_batch_size
        remaining -= train_batch_size + val_batch_size
        counter += 1
        if train_or_val == "both": 
            yield train_imgs, train_masks, train_index, val_imgs, val_masks, val_index
        if train_or_val ==  "train": 
            yield (train_imgs, train_masks)
        if train_or_val == "val": 
            yield (val_imgs, val_masks)


def test_data_generator(file, img_rows, img_cols, iter=1):
    for imgs, masks, index, val_imgs, val_masks, val_index in \
            train_val_data_generator(file, train_batch_size=1, val_batch_size=0, img_rows=img_rows, img_cols=img_cols, iter=iter):
        yield imgs, masks, index


def train_val_generator(file, img_rows, img_cols, batch_size=100, train_size=1, val_size=1, train_or_val="both", iter=1000, shuffle=True):
    f = h5py.File(file, 'r')
    f.visititems(list_all_patients)
    p_list = patient_group_dict.keys()
    while True: 
        remaining = len(p_list)
        train_counter = 0
        val_counter = train_counter + train_size
        if train_size + val_size > remaining:
            print 'Not enough data!'
            raise StopIteration
        counter = 0
        while train_size + val_size < remaining and counter < iter:
            p_sublist = p_list[train_counter:(train_counter + train_size)]
            train_imgs = np.array([]).reshape((0, 1, img_rows, img_cols))
            train_masks = np.array([]).reshape((0, 1, img_rows, img_cols))
            train_index = np.array([]).reshape((0, 4))
            for p in p_sublist:
                imgs, masks, indices = load_data_from_hdf5(
                    file, p, patient_group_dict)
                train_imgs = np.vstack((train_imgs, imgs))
                train_masks = np.vstack((train_masks, masks))
                train_index = np.vstack((train_index, indices))
            if shuffle: 
                ix = np.arange(train_imgs.shape[0])
                np.random.shuffle(ix)
                train_imgs = train_imgs[ix,:,:,:]
                train_masks = train_masks[ix,:,:,:]
                train_index = train_index[ix,:]
            val_imgs = np.array([]).reshape((0, 1, img_rows, img_cols))
            val_masks = np.array([]).reshape((0, 1, img_rows, img_cols))
            val_index = np.array([]).reshape((0, 4))
            if val_size > 0:
                p_sublist = p_list[val_counter:(val_counter + val_size)]
                imgs_len = 0
                for p in p_sublist:
                    imgs, masks, indices = load_data_from_hdf5(
                        file, p, patient_group_dict)
                    imgs_len += imgs.shape[0]
                    val_imgs = np.vstack((val_imgs, imgs))
                    val_masks = np.vstack((val_masks, masks))
                    val_index = np.vstack((val_index, indices))
                val_counter = train_counter + train_size
                if shuffle: 
                    ix = np.arange(val_imgs.shape[0])
                    np.random.shuffle(ix)
                    val_imgs = val_imgs[ix,:,:,:]
                    val_masks = val_masks[ix,:,:,:]
                    val_index = val_index[ix,:]
            train_counter = train_counter + train_size + val_size
            remaining -= train_size + val_size
            if train_or_val == 'train': 
                print " Now at train Iteration", counter
            if train_or_val == 'val': 
                print " Now at val Iteration", counter
            counter += 1

            train_img_size = train_imgs.shape[0]
            val_img_size = val_imgs.shape[0]
            if train_or_val == "both": 
                yield train_imgs, train_masks, train_index, val_imgs, val_masks, val_index
            if train_or_val ==  "train": 
                start_index = 0
                end_index = start_index + batch_size
                batch = 0
                while end_index <= train_img_size: 
                    train_img_batch = train_imgs[start_index:end_index, :, :, :]
                    train_mask_batch = train_masks[start_index:end_index, :, :, :]
                    start_index = end_index
                    end_index = start_index + batch_size                        
                    print " now yielding train batch", batch
                    batch += 1 
                    yield (train_img_batch, train_mask_batch)
            if train_or_val == "val": 
                start_index = 0
                end_index = start_index + batch_size
                batch = 0
                while end_index <= val_img_size and val_img_size > 0: 
                    val_img_batch = val_imgs[start_index:end_index, :, :, :]
                    val_mask_batch = val_masks[start_index:end_index, :, :, :]
                    start_index = end_index
                    end_index = start_index + batch_size
                    print " now yielding val batch", batch
                    batch += 1 
                    yield (val_img_batch, val_mask_batch)


