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
import argparse
import sys
from data_utils import train_val_data_generator, test_data_generator


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

# read data from src_dirs, write to des_file in HDF5 binary; Scale image & mask to reduced_size; if augmentation > 1, generate # of 'augmentation' images for each ggo image. 


def create_data(src_dirs, des_dir, normalization = True, reduced_size = None, augmentation = 1):
    if normalization and reduced_size != None: 
        img_rows = reduced_size[0]
        img_cols = reduced_size[1]
    else: 
        img_rows = 512
        img_cols = 512
    des_file = os.path.join(des_dir, "train_data.hdf5")
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
                    img_pixel = cv2.resize(img_pixel, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
                    has_ggo = False
                    if hasattr(img, 'BodyPartExamined'):
                        body_part = img.BodyPartExamined
                    else:
                        body_part = None
                    if mask_path != None:
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
                        # mask = np.array([mask])
                        has_ggo = True
                    else:
                        mask = np.full((img_rows, img_cols), 255, dtype=np.uint8)
                        # mask = np.array([mask])
                    counter += 1
                    index = np.array(
                        [p, str(counter), str(has_ggo), str(body_part)])
                    imgs.append([img_pixel])
                    masks.append([mask])
                    indices.append(index)
                    if augmentation > 1 and has_ggo:
                        for i in range(augmentation):
                            img_tf, mask_tf = transform(img_pixel,mask)
                            counter += 1
                            index = np.array([p, str(counter), str(has_ggo), str(body_part)])
                            imgs.append([img_tf])
                            masks.append([mask_tf])
                            indices.append(index)
                    if counter % 100 == 0:
                        print 'Done: {0} images'.format(counter)
 
        if normalization: 
            m = np.mean(imgs).astype(np.float32)
            imgs -= m
            st = np.std(imgs).astype(np.float32)
            imgs /= st
            m = np.mean(masks).astype(np.float32)
            masks -= m
            st = np.std(masks).astype(np.float32)
            masks /= (st+1e-6)
            print 'std --- '+str(st)
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


# read data from train_dirs, write to dest_dir in HDF5 binary
def create_train_data_single(img_dir, mask_dir, dest_dir):
    imgs = []
    masks = []
    indices = []
    leaf_dir_name = os.path.basename(os.path.normpath(img_dir))
    dir_name = os.path.join(dest_dir, leaf_dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # train_imgs_save_to = os.path.join(dir_name, 'train_imgs.npy')
    # train_masks_save_to = os.path.join(dir_name, 'train_masks.npy')
    # train_index_save_to = os.path.join(dir_name, 'train_index.npy')
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
                if img_pixel.shape[0] != 512 or img_pixel.shape[1] != 512:
                    img_pixel = cv2.resize(
                            img_pixel, (512, 512), interpolation=cv2.INTER_CUBIC)
                has_ggo = False
                if hasattr(img, 'BodyPartExamined'):
                    body_part = img.BodyPartExamined
                else:
                    body_part = None
                if mask_path != None:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = np.array([mask])
                    has_ggo = True
                else:
                    mask = np.full((512, 512), 255, dtype=np.uint8)
                    mask = np.array([mask])
                counter += 1
                index = np.array(
                        [p, str(counter), str(has_ggo), str(body_part)])
                imgs.append([img_pixel])
                masks.append(mask)
                indices.append(index)
                if counter % 100 == 0:
                    print 'Done: {0} images'.format(counter)
    imgs = np.array(imgs)
    np.reshape(imgs, (len(imgs), 1, 512, 512))
    masks = np.array(masks)
    indices = np.array(indices)

    print 'Loading done.'

    #np.save(train_imgs_save_to, imgs)
    #np.save(train_masks_save_to, masks)
    #np.save(train_index_save_to, imgs_id)
    # print 'Saving to .py files done'
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


def transform_train_data_generator(file, train_batch_size = 5, normalization = True, reduced_size=None, augmentationfactor = 1):
    f = h5py.File(file, 'r')
    f.visititems(list_all_patients)
    p_list = patient_group_dict.keys()
    remaining = len(p_list)
    counter = 0
    if train_batch_size > remaining:
        print 'Not enough data!'
    while train_batch_size < remaining:
        p_sublist = p_list[counter:(counter + train_batch_size)]
        train_imgs = np.array([]).reshape((0, 1, 512, 512))
        train_masks = np.array([]).reshape((0, 1, 512, 512))
        train_index = np.array([]).reshape((0, 4))
        for p in p_sublist:
            imgs, masks, indices = load_data_from_hdf5(
                file, p, patient_group_dict)
            train_imgs = np.vstack((train_imgs, imgs))
            train_masks = np.vstack((train_masks, masks))
            train_index = np.vstack((train_index, indices))
        counter = counter + train_batch_size
        remaining -= train_batch_size 

        data_shape = train_imgs.shape
        train_imgs_tf = np.ndarray((data_shape[0]*augmentationfactor, data_shape[1], data_shape[2], data_shape[3]))
        train_masks_tf = np.ndarray((data_shape[0]*augmentationfactor, data_shape[1], data_shape[2], data_shape[3]))
        count = 0
        for i in range(data_shape[0]): 
            for j in range(augmentationfactor): 
                train_imgs_tf[count][0], train_masks_tf[count][0] = transform(train_imgs[count][0],train_masks[count][0])
#                plt.imshow(train_imgs[count][0]-train_imgs[count][0])
#                plt.show()
                count += 1
        train_imgs_p, m, st = preprocessing_imgs(train_imgs_tf, reduced_size)
        train_masks_p = preprocessing_masks(train_masks_tf, reduced_size)
        yield train_imgs_p, train_imgs_p

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

    #def_image = def_image[10:-10, 10:-10]
    #def_mask = def_mask[10:-10, 10:-10]
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
    parser.add_argument('--output_dir', metavar='output_dir', nargs='?',
                        help='output directory', required=True)

    return parser.parse_args()


if __name__ == '__main__':
    img_rows = 128 
    img_cols = 128 

    args = parse_options()
    # train_imgs, train_masks, train_index = create_train_data(args.img_dir, args.mask_dir, args.output_dir)
    # print train_imgs.shape, train_masks.shape

    train_imgs, train_masks, train_index = create_data(args.input_dirs, args.output_dir, normalization=True, reduced_size=[img_rows, img_cols], augmentation=20)

    # print train_imgs.shape, train_masks.shape
    # train_val_data_generator('../../../train_data.hdf5')
    # train_imgs, train_masks, train_index = load_train_data("../../../")
    # train_imgs_p, m, st = preprocessing_imgs (train_imgs,reduced_size=(128,128))

    # train_imgs_p, m, st = preprocessing_imgs(train_imgs, reduced_size=(128,128))
    # print train_imgs_p.shape, m, st
    # train_masks_p = preprocessing_masks(train_masks,reduced_size=(128,128))
    # plt.imshow(train_imgs[0])
    # plt.imshow(train_masks[0,0])
    # plt.show()
