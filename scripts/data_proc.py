#!/usr/bin/env python2.7

import os
import dicom
import fnmatch
import re
import os.path
import cv2
import numpy as np
from dicom.filereader import InvalidDicomError
import logging
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy.random as random
import skimage.transform as tf
import h5py
import ConfigParser
import argparse
import sys
import scipy.ndimage as ndi
from data_utils import train_val_data_generator, test_data_generator
import matplotlib.pyplot as plt
from scipy import linalg

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# match images and masks
logging_handler_out = logging.StreamHandler(sys.stdout)
logger.addHandler(logging_handler_out)


def get_img_mask_dict(img_dir, mask_dir):
    logger.info("processing {0}".format(img_dir))
    img_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(
        img_dir) for f in fnmatch.filter(files, "IMG*")]
    img_dict = {}
    for img in img_list:
        try:
            f = dicom.read_file(img)
        except IOError:
            print 'No such file'
        except InvalidDicomError:
            print 'Invalid Dicom file {0}'.format(img)
        patientID = f.PatientID
        if hasattr(f, 'SeriesNumber'):
            seriesNumber = f.SeriesNumber
        else:
            seriesNumber = 'ELSE'
        if patientID not in img_dict:
            img_dict[patientID] = {}
            img_dict[patientID]['age'] = f.PatientsAge
            img_dict[patientID]['sex'] = f.PatientsSex
            img_dict[patientID]['img_series'] = {}
        if seriesNumber not in img_dict[patientID]['img_series']:
            img_dict[patientID]['img_series'][seriesNumber] = {}
            img_dict[patientID]['img_series'][seriesNumber][
                'path'] = re.search(r'(.*SRS\d{1,})(/IMG\d{1,})', img).group(1)
            img_dict[patientID]['img_series'][seriesNumber]['imgs'] = []
            img_dict[patientID]['img_series'][seriesNumber]['masks'] = []
            img_dict[patientID]['img_series'][seriesNumber]['counter_img'] = 0
            img_dict[patientID]['img_series'][seriesNumber]['counter_mask'] = 0

        img_dict[patientID]['img_series'][seriesNumber]['imgs'].append(img)
        img_dict[patientID]['img_series'][seriesNumber]['counter_img'] += 1
        path = mask_dir + '/' + re.search(r'(.*DICOMDAT/)(SDY.*)', img_dict[patientID][
            'img_series'][seriesNumber]['path']).group(2)
        mask_name = re.search(
            r'(.*SRS\d{1,}/)(IMG\d{1,})', img).group(2) + '-mask.tif'
        mask_path = os.path.join(path, mask_name)
        if os.path.isfile(mask_path):
            img_dict[patientID]['img_series'][
                seriesNumber]['masks'].append(mask_path)
            img_dict[patientID]['img_series'][
                seriesNumber]['counter_mask'] += 1
        else:
            img_dict[patientID]['img_series'][
                seriesNumber]['masks'].append(None)

    total_img = 0
    total_case = 0
    for p in img_dict.keys():
        print '#' * 30
        print 'patient id : ' + p
        for s in img_dict[p]['img_series'].keys():
            total_img += img_dict[p]['img_series'][s]['counter_img']
            total_case += img_dict[p]['img_series'][s]['counter_mask']
            print 'series number: ' + str(s) + ' || dir_path: ' + img_dict[p]['img_series'][s]['path'] + " || number of images: " + str(img_dict[p]['img_series'][s]['counter_img']) + " || number of masks: " + str(img_dict[p]['img_series'][s]['counter_mask'])
        img_dict[p]['total_img'] = total_img
        img_dict[p]['total_case'] = total_case
    print 'Total number of patients: ' + str(len(img_dict.keys()))
    print 'Total number of images: ' + str(total_img)
    print 'Total number of cases: ' + str(total_case)

    return img_dict

# read data from src_dirs, write to des_file in HDF5 binary; Scale image &
# mask to reduced_size; if augmentation > 1, generate # of 'augmentation'
# images for each ggo image.


def create_data(src_dirs, des_file, original_size, normalization=True, reduced_size=None, ggo_aug=50, crop=False, cropped_size=None, label_smoothing=1e-4):
    if crop and cropped_size:
        img_rows = int(cropped_size[0])
        img_cols = int(cropped_size[1])
    elif reduced_size is not None:
        img_rows = int(reduced_size[0])
        img_cols = int(reduced_size[1])
    else:
        img_rows = int(original_size[0])
        img_cols = int(original_size[1])
    f = h5py.File(des_file, "w")
    for img_dir in src_dirs:
        imgs = []
        masks = []
        indices = []
        mask_dir = img_dir + '_MASK'
        img_dict = get_img_mask_dict(img_dir, mask_dir)
        counter = 0
        for p in img_dict.keys():
            for s in img_dict[p]['img_series'].keys():
                for img_path, mask_path in zip(img_dict[p]['img_series'][s]['imgs'], img_dict[p]['img_series'][s]['masks']):
                    try:
                        img = dicom.read_file(img_path)
                    except IOError:
                        print 'No such file'
                    except InvalidDicomError:
                        print 'Invalid Dicom file {0}'.format(img)
                    img_pixel = np.array(img.pixel_array)
                    if crop:
                        img_pixel = img_pixel[90:450, 40:470]
                    if reduced_size != None:
                        img_pixel = cv2.resize(
                            img_pixel, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
                    has_ggo = False
                    if hasattr(img, 'BodyPartExamined'):
                        body_part = img.BodyPartExamined
                    else:
                        body_part = None
                    if mask_path != None:
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if crop:
                            mask = mask[90:450, 40:470]
                        if reduced_size != None:
                            mask = cv2.resize(
                                mask, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
                        has_ggo = True
                    else:
                        mask = np.full((img_rows, img_cols),
                                       255, dtype=np.uint8)
                        if label_smoothing != None:
                            index = np.random.choice([False, True], (img_rows, img_cols), replace=True, p=[
                                                     1 - label_smoothing, label_smoothing])
                            mask[index] = 0
                    counter += 1
                    index = np.array(
                        [p, str(counter), str(has_ggo), str(body_part)])
                    imgs.append([img_pixel])
                    masks.append([mask])
                    indices.append(index)

                    if ggo_aug > 1 and has_ggo:
                        for i in range(ggo_aug):
                            img_tf, mask_tf = transform(img_pixel, mask)
                            counter += 1
                            index = np.array(
                                [p, str(counter), str(has_ggo), str(body_part)])
                            imgs.append([img_tf])
                            masks.append([mask_tf])
                            indices.append(index)
                    if counter % 1000 == 0:
                        print 'Done: {0} images'.format(counter)

        if normalization:
            m = np.mean(imgs).astype(np.float32)
            imgs -= m
            st = np.std(imgs).astype(np.float32)
            imgs /= st
            masks = np.asarray(masks)
            masks /= 255
            masks[masks < 0.5] = 0
            masks[masks >= 0.5] = 1
            masks.astype(int)
        imgs = np.array(imgs)
        np.reshape(imgs, (len(imgs), 1, img_rows, img_cols))
        masks = np.array(masks)
        np.reshape(masks, (len(masks), 1, img_rows, img_cols))
        indices = np.array(indices)
        grp = f.create_group(img_dir)
        dset = grp.create_dataset("imgs", data=imgs)
        dset = grp.create_dataset("masks", data=masks)
        dset = grp.create_dataset("indices", data=indices)
    print 'Loading done.'
    print 'Saving to h5py files done'
    return imgs, masks, indices


def load_train_data(data_path):
    file_paths = get_filepaths(data_path)
    for file_path in file_paths:
        train_imgs = np.load(os.path.join(file_path, 'train_imgs.npy'))
        train_masks = np.load(os.path.join(file_path, 'train_masks.npy'))
        train_index = np.load(os.path.join(file_path, 'train_index.npy'))
    return train_imgs, train_masks, train_index

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

def preprocessing_imgs(train_imgs, reduced_size=None):
    # resizing
    if reduced_size is not None:
        train_imgs_p = np.ndarray(
            (train_imgs.shape[0], 1) + reduced_size, dtype=np.float32)
        for i in range(train_imgs.shape[0]):
            train_imgs_p[i, 0] = cv2.resize(train_imgs[i][0], (reduced_size[1], reduced_size[
                                            0]), interpolation=cv2.INTER_CUBIC)  # INVERSE ORDER! cols,rows
    else:
        train_imgs_p = train_imgs.astype(np.float32)

    # ZMUV normalization
    m = np.mean(train_imgs_p).astype(np.float32)
    train_imgs_p -= m
    st = np.std(train_imgs_p).astype(np.float32)
    train_imgs_p /= st

    return train_imgs_p, m, st


def transform(image, mask):  # translate, shear, stretch, flips?
    rows, cols = image.shape

    angle = random.uniform(-1.5, 1.5)
    center = (rows / 2 - 0.5 + random.uniform(-50, 50),
              cols / 2 - 0.5 + random.uniform(-50, 50))
    def_image = tf.rotate(image, angle=angle, center=center,
                          clip=True, preserve_range=True, order=5)
    if (mask - mask[0][0] == 0).all():
        def_mask = mask
    else:
        def_mask = tf.rotate(mask, angle=angle, center=center,
                             clip=True, preserve_range=True, order=5)
    alpha = random.uniform(0, 5)
    sigma = random.exponential(scale=5) + 2 + alpha**2
    def_image, def_mask = elastic_transform(def_image, def_mask, alpha, sigma)
    return def_image, def_mask

# sigma: variance of filter, fixes homogeneity of transformation
#    (close to zero : random, big: translation)


def elastic_transform(image, mask, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

       Code taken from https://gist.github.com/fmder/e28813c1e8721830ff9c
       slightly modified
    """
    min_im = np.min(image)
    max_im = np.max(image)

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0)
    dx = dx / np.max(dx) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0)
    dy = dy / np.max(dy) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    image_tfd = map_coordinates(image, indices, order=3).reshape(shape)
    image_tfd[image_tfd > max_im] = max_im
    image_tfd[image_tfd < min_im] = min_im

    min_im = np.min(mask)
    max_im = np.max(mask)
    mask_tfd = map_coordinates(mask, indices, order=3).reshape(shape)
    mask_tfd[mask_tfd > max_im] = max_im
    mask_tfd[mask_tfd < min_im] = min_im

    return image_tfd, mask_tfd


def preprocessing_masks(train_masks, reduced_size=None):
    # resizing
    if reduced_size is not None:
        train_masks_p = np.ndarray((train_masks.shape[0], train_masks.shape[
                                   1]) + reduced_size, dtype=np.uint8)
        for i in range(train_masks.shape[0]):
            if np.min(train_masks[i, 0]) < 255:
                train_masks_p[i, 0] = cv2.resize(train_masks[i, 0], (reduced_size[
                                                 1], reduced_size[0]), interpolation=cv2.INTER_LINEAR)
            else:
                train_masks_p[i, 0] = np.full(
                    (reduced_size[0], reduced_size[1]), 255, dtype=np.uint8)
    else:
        train_masks_p = train_masks
    # to [0,1]
    train_masks_p = (train_masks_p / 255).astype(np.uint8)
    return train_masks_p


def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    :param directory:
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        # print directories
        for directory in directories:
            # Join the two strings in order to form the full filepath.
            # filepath = os.path.join(root, filename)
            # print root
            # Add it to the list.
            file_path = os.path.join(root, directory)
            file_paths.append(file_path)
            logger.info(file_path)

    return file_paths  # Self-explanatory.


def parse_options():
    parser = argparse.ArgumentParser(description='Process DICOM Images.')
    parser.add_argument('--input_dirs', metavar='input_dirs', nargs='+',
                        help='input directory for source images and masks', required=True)
    parser.add_argument('--output_file', metavar='output_file', nargs='?',
                        help='output file', required=True)
    parser.add_argument('--config_file', metavar='config_file', nargs='?',
                        help='config file', required=True)

    return parser.parse_args()

def create_data_2(src_dirs, des_file, original_size, normalization=True, whitening=False, reduced_size=None, ggo_aug=50, crop=False, cropped_size=None, label_smoothing=1e-4):
    if crop and cropped_size:
        img_rows = int(cropped_size[0])
        img_cols = int(cropped_size[1])
    elif reduced_size is not None:
        img_rows = int(reduced_size[0])
        img_cols = int(reduced_size[1])
    else:
        img_rows = int(original_size[0])
        img_cols = int(original_size[1])
    f = h5py.File(des_file, "w")
    for img_dir in src_dirs:
        imgs = []
        masks = []
        indices = []
        mask_dir = img_dir + '_MASK'
        img_dict = get_img_mask_dict(img_dir, mask_dir)
        counter = 0
        for p in img_dict.keys():
            for s in img_dict[p]['img_series'].keys():
                for img_path, mask_path in zip(img_dict[p]['img_series'][s]['imgs'], img_dict[p]['img_series'][s]['masks']):
                    try:
                        img = dicom.read_file(img_path)
                    except IOError:
                        print 'No such file'
                    except InvalidDicomError:
                        print 'Invalid Dicom file {0}'.format(img)
                    img_pixel = np.array(img.pixel_array)
                    if crop:
                        img_pixel = img_pixel[90:450, 40:470]
                    if reduced_size != None:
                        img_pixel = cv2.resize(
                            img_pixel, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
                    has_ggo = False
                    if hasattr(img, 'BodyPartExamined'):
                        body_part = img.BodyPartExamined
                    else:
                        body_part = None
                    if mask_path != None:
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if crop:
                            mask = mask[90:450, 40:470]
                        if reduced_size != None:
                            mask = cv2.resize(
                                mask, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
                        has_ggo = True
                    else:
                        mask = np.full((img_rows, img_cols),
                                       255, dtype=np.uint8)
                        if label_smoothing != None:
                            index = np.random.choice([False, True], (img_rows, img_cols), replace=True, p=[
                                                     1 - label_smoothing, label_smoothing])
                            mask[index] = 0
                    counter += 1
                    index = np.array(
                        [p, str(counter), str(has_ggo), str(body_part)])
                    imgs.append([img_pixel])
                    masks.append([mask])
                    indices.append(index)
                    if ggo_aug > 1 and has_ggo:
                        for i in range(ggo_aug):
                            img_tf, mask_tf = random_transform(img_pixel, mask, 
                                rotation_range=90, height_shift_range=0, width_shift_range=0, shear_range=1, zoom_range=(1,1), horizontal_flip=True, vertical_flip=True)
                            counter += 1
                            index = np.array(
                                [p, str(counter), str(has_ggo), str(body_part)])
                            imgs.append([img_tf])
                            masks.append([mask_tf])
                            indices.append(index)
                            #plt.imshow(img_tf)
                            #plt.gray()
                            #plt.show()
                            #plt.imshow(mask_tf)
                            #plt.gray()
                            #plt.show()
                    if counter % 1000 == 0:
                        print 'Done: {0} images'.format(counter)

        if normalization:
            m = np.mean(imgs).astype(np.float32)
            imgs -= m
            st = np.std(imgs).astype(np.float32)
            imgs /= (st + 1e-6)
            masks = np.asarray(masks)
            masks /= 255
            masks[masks < 0.5] = 0
            masks[masks >= 0.5] = 1
            masks.astype(int)
        if whitening: 
            remaining = imgs.shape[0]
            idx = 0
            while remaining > 100: 
                plt.imshow(imgs[idx,0,:,:])
                plt.gray()
                plt.show()
                imgs[idx:(idx+100),:,:,:] = zca_whitening(imgs[idx:(idx+100),:,:,:])
                plt.imshow(imgs[idx,0,:,:])
                plt.gray()
                plt.show()
                idx += 100
                remaining -= 100
            imgs[idx:,:,:,:] = zca_whitening(imgs[idx:,:,:,:])
        imgs = np.array(imgs)
        np.reshape(imgs, (len(imgs), 1, img_rows, img_cols))
        masks = np.array(masks)
        np.reshape(masks, (len(masks), 1, img_rows, img_cols))
        indices = np.array(indices)
        grp = f.create_group(img_dir)
        dset = grp.create_dataset("imgs", data=imgs)
        dset = grp.create_dataset("masks", data=masks)
        dset = grp.create_dataset("indices", data=indices)
    print 'Loading done.'
    print 'Saving to h5py files done'
    return imgs, masks, indices

def random_transform(image, mask, rotation_range=90, height_shift_range=0, width_shift_range=0, shear_range=1, zoom_range=(2,2), horizontal_flip=True, vertical_flip=True):  
    h, w = image.shape
    x = np.expand_dims(image, axis=0)
    y = np.expand_dims(mask, axis=0)
    img_row_index = 1
    img_col_index = 2

    if rotation_range:
        theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
    if height_shift_range:
        tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_index]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    if shear_range:
        shear = np.random.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else: 
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix)
    y = apply_transform(y, transform_matrix)
    if horizontal_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, 2)
            y = flip_axis(y, 2)
    if vertical_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, 1)
            y = flip_axis(y, 1)
    return x[0,:,:], y[0,:,:]


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix, 
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def zca_whitening(X): 
    flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    sigma = np.dot(flatX.T, flatX) / flatX.shape[0]
    U, S, V = linalg.svd(sigma)
    principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
    XW = np.zeros(X.shape)
    for i in range(X.shape[0]): 
        x = X[i,:,:,:]
        flatx = np.reshape(x, (x.size)) 
        whitex = np.dot(flatx, principal_components)
        XW[i,:,:,:] = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return XW

if __name__ == '__main__':
    args = parse_options()

    config = ConfigParser.ConfigParser()
    config.read(args.config_file)
    data_config = dict(config.items('config'))

    img_rows = int(data_config.get('img_rows', 512))
    img_cols = int(data_config.get('img_cols', 512))

    reduced_img_rows = data_config.get('reduced_img_rows', None)
    reduced_img_cols = data_config.get('reduced_img_cols', None)

    cropped_img_rows = data_config.get('cropped_img_rows', None)
    cropped_img_cols = data_config.get('cropped_img_cols', None)

    ggo_aug = int(data_config.get('ggo_aug', 100))

    reduced_size = None
    if reduced_img_rows and reduced_img_cols:
        reduced_size = [int(reduced_img_rows), int(reduced_img_cols)]

    cropped_size = None
    if cropped_img_rows and cropped_img_cols:
        reduced_size = [int(cropped_img_rows), int(cropped_img_cols)]

    train_imgs, train_masks, train_index = create_data_2(
        args.input_dirs, args.output_file, original_size=[
            img_rows, img_cols], normalization=True, whitening=False, reduced_size=reduced_size, ggo_aug=ggo_aug, crop=False,
        cropped_size=cropped_size)
