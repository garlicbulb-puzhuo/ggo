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


def train_val_data_generator(file, img_rows, img_cols, train_batch_size=5, val_batch_size=2):
    f = h5py.File(file, 'r')
    f.visititems(list_all_patients)
    p_list = patient_group_dict.keys()
    remaining = len(p_list)
    train_counter = 0
    val_counter = train_counter + train_batch_size
    if train_batch_size + val_batch_size > remaining:
        print 'Not enough data!'
    while train_batch_size + val_batch_size < remaining:
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
        p_sublist = p_list[val_counter:(val_counter + val_batch_size)]
        val_imgs = np.array([]).reshape((0, 1, img_rows, img_cols))
        val_masks = np.array([]).reshape((0, 1, img_rows, img_cols))
        val_index = np.array([]).reshape((0, 4))
        imgs_len = 0
        for p in p_sublist:
            imgs, masks, indices = load_data_from_hdf5(
                file, p, patient_group_dict)
            imgs_len += imgs.shape[0]
            val_imgs = np.vstack((val_imgs, imgs))
            val_masks = np.vstack((val_masks, masks))
            val_index = np.vstack((val_index, indices))
        val_imgs = np.array(val_imgs)
        train_counter = train_counter + train_batch_size + val_batch_size
        val_counter = train_counter + train_batch_size
        remaining -= train_batch_size + val_batch_size
        yield train_imgs, train_masks, train_index, val_imgs, val_masks, val_index


def test_data_generator(file, img_rows, img_cols, iter=1):
    for imgs, masks, index, val_imgs, val_masks, val_index in \
            train_val_data_generator(file, train_batch_size=1, val_batch_size=0, img_rows=img_rows, img_cols=img_cols, iter=iter):
        yield imgs, masks, index
