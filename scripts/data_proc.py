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


logger = logging.getLogger()


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

#train_dirs = ['../../../CA1','../../../CA2','../../../CA3']


def create_train_data(train_dirs, dest_dir):
    imgs = []
    masks = []
    imgs_id = []
    train_imgs_save_to = os.path.join(dest_dir, 'train_imgs.npy')
    train_masks_save_to = os.path.join(dest_dir, 'train_masks.npy')
    train_index_save_to = os.path.join(dest_dir, 'train_index.npy')
    for img_dir in train_dirs:
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
                    if mask_path != None:
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        mask = np.array([mask])
                    else:
                        mask = np.full((512, 512), 255, dtype=np.uint8)
                        mask = np.array([mask])
                    counter += 1
                    img_id = [p, counter]
                    imgs.append(img_pixel)
                    masks.append(mask)
                    imgs_id.append(img_id)
                    if counter % 100 == 0:
                        print 'Done: {0} images'.format(counter)
    imgs = np.array(imgs)
    masks = np.array(masks)
    print 'Loading done.'

    np.save(train_imgs_save_to, imgs)
    np.save(train_masks_save_to, masks)
    np.save(train_index_save_to, imgs_id)
    print 'Saving to .py files done'
    return imgs, masks, imgs_id


def load_train_data(data_path):
    imgs_train = np.load(data_path + 'train_imgs.npy')
    imgs_mask_train = np.load(data_path + 'train_masks.npy')
    imgs_train_id = np.load(data_path + 'train_index.npy')
    return imgs_train, imgs_mask_train, imgs_train_id


def preprocessing_imgs(train_imgs, reduced_size=None):
    # resizing
    if reduced_size is not None:
        train_imgs_p = np.ndarray(
            (train_imgs.shape[0], 1) + reduced_size, dtype=np.float32)
        for i in range(train_imgs.shape[0]):
            train_imgs_p[i, 0] = cv2.resize(train_imgs[i], (reduced_size[1], reduced_size[
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
        ['../../../CA1'], '../../../')
    # print train_imgs.shape, train_masks.shape
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
