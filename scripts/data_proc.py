import os
import dicom
import fnmatch
import re
import os.path
import cv2
import numpy as np
from dicom.filereader import InvalidDicomError
import logging
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy.random as random
import skimage.transform as tf
import h5py


logger = logging.getLogger()

# match images and masks


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
        # print mask_path
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

# read data from train_dirs, write to dest_dir in HDF5 binary


def create_train_data(train_dirs, dest_dir):
    #    imgs = []
    #    masks = []
    #    imgs_id = []
    #    train_imgs_save_to = os.path.join(dest_dir, 'train_imgs.npy')
    #    train_masks_save_to = os.path.join(dest_dir, 'train_masks.npy')
    #    train_index_save_to = os.path.join(dest_dir, 'train_index.npy')
    f = h5py.File(os.path.join(dest_dir, "train_data.hdf5"), "w")
#    f = h5py.File("train_data.hdf5", "w")

    for img_dir in train_dirs:
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
        grp = f.create_group(img_dir)
        dset = grp.create_dataset("imgs", data=imgs)
        dset = grp.create_dataset("masks", data=masks)
        dset = grp.create_dataset("indices", data=indices)
    #imgs = np.array(imgs)
    #masks = np.array(masks)
    print 'Loading done.'

    #np.save(train_imgs_save_to, imgs)
    #np.save(train_masks_save_to, masks)
    #np.save(train_index_save_to, imgs_id)
    # print 'Saving to .py files done'
    print 'Saving to h5py files done'
    return masks, masks, indices


def load_train_data_from_npy(data_path):
    train_imgs = np.load(data_path + 'train_imgs.npy')
    train_masks = np.load(data_path + 'train_masks.npy')
    train_index = np.load(data_path + 'train_index.npy')
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


def train_val_data_generator(file, train_batch_size=5, val_batch_size=2, normalization=True, reduced_size=None):
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
        train_imgs = np.array([]).reshape((0, 1, 512, 512))
        train_masks = np.array([]).reshape((0, 1, 512, 512))
        train_index = np.array([]).reshape((0, 4))
        for p in p_sublist:
            imgs, masks, indices = load_data_from_hdf5(
                file, p, patient_group_dict)
            train_imgs = np.vstack((train_imgs, imgs))
            train_masks = np.vstack((train_masks, masks))
            train_index = np.vstack((train_index, indices))
        p_sublist = p_list[val_counter:(val_counter + val_batch_size)]
        val_imgs = np.array([]).reshape((0, 1, 512, 512))
        val_masks = np.array([]).reshape((0, 1, 512, 512))
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
        if normalization:
            print train_imgs.shape
            print train_counter
            train_imgs_p, m, st = preprocessing_imgs(train_imgs, reduced_size)
            train_masks_p = preprocessing_masks(train_masks, reduced_size)
            val_imgs_p, m_val, st_val = preprocessing_imgs(
                val_imgs, reduced_size)
            val_masks_p = preprocessing_masks(val_masks, reduced_size)
            yield train_imgs_p, train_masks_p, train_index, val_imgs_p, val_masks_p, val_index, m, st
        else:
            yield train_imgs, train_masks, train_index, val_imgs, val_masks, val_index


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


def transform(image):  # translate, shear, stretch, flips?
    rows, cols = image.shape

    angle = random.uniform(-1.5, 1.5)
    center = (rows / 2 - 0.5 + random.uniform(-50, 50),
              cols / 2 - 0.5 + random.uniform(-50, 50))
    def_image = tf.rotate(image, angle=angle, center=center,
                          clip=True, preserve_range=True, order=5)

    alpha = random.uniform(0, 5)
    sigma = random.exponential(scale=5) + 2 + alpha**2
    def_image = elastic_transform(def_image, alpha, sigma)

    def_image = def_image[10:-10, 10:-10]

    return def_image

# sigma: variance of filter, fixes homogeneity of transformation
#    (close to zero : random, big: translation)


def elastic_transform(image, alpha, sigma, random_state=None):
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

    return image_tfd


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


if __name__ == '__main__':
    #img_dict = get_img_mask_dict('../../../CA1', '../../../CA1_MASK')

    train_imgs, train_masks, train_index = create_train_data(
        ['../../../CA1', '../../../CA2', '../../../CA3'], '../../../')
    # print train_imgs.shape, train_masks.shape
    # train_val_data_generator('../../../train_data.hdf5')
    #train_imgs, train_masks, train_index = load_train_data("../../../")
    #train_imgs_p, m, st = preprocessing_imgs (train_imgs,reduced_size=(128,128))
    # print train_imgs_p.shape, m, st
    #train_masks_p = preprocessing_masks(train_masks,reduced_size=(128,128))
    # plt.imshow(train_imgs[0])
    # plt.imshow(train_masks[0,0])
    # plt.show()
    # plt.imshow(train_imgs_p[0,0])
    # plt.imshow(train_masks_p[0,0])
    # plt.show()
    #ts = transform(train_masks_p[0,0])
    # plt.imshow(ts)
    # plt.show()
