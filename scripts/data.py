#!/usr/bin/env python2.7

import argparse
import os
import logging
import sys
import shutil
from data_proc import get_img_mask_dict
import re
import json


logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging_handler_out = logging.StreamHandler(sys.stdout)
logger.addHandler(logging_handler_out)


def get_parser():
    parser = argparse.ArgumentParser(description='Merge DICOM Images.')
    parser.add_argument('-i', '--input-dir', nargs='?', required=True,
                        help='input directory for source images')
    parser.add_argument('-m', '--mask-input-dir', nargs='?', required=True,
                        help='input directory for mask images')
    parser.add_argument('-o', '--output-dir', nargs='?', required=True,
                        help='output directory')

    return parser.parse_args()


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
        for filename in files:
            # check DICOMAT file to see if the folder is eligible
            if True:
                break

            # Join the two strings in order to form the full filepath.
            # filepath = os.path.join(root, filename)
            # print root
            # Add it to the list.
            file_paths.append(os.path.join(root, filename))

    return file_paths  # Self-explanatory.


def save(img_file, directory):
    shutil.copy2(img_file, directory)

if __name__ == '__main__':
    args = get_parser()
    img_dir = os.path.abspath(os.path.join(args.input_dir, 'DICOMDAT'))
    mask_dir = os.path.abspath(args.mask_input_dir)
    file_paths = get_filepaths(img_dir)
    img_mask_dict = get_img_mask_dict(img_dir, mask_dir)

    logger.info("root path {0}".format(img_dir))

    output_path = os.path.abspath(args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for patient_id, value in img_mask_dict.items():
        logger.info("processing patient {0}".format(patient_id))
        img_series_dict = value['img_series']
        patient_dir = os.path.join(output_path, patient_id)

        for k, v in img_series_dict.items():
            sub_dir = re.sub(
                '[/]', '_', re.search(r'(.*DICOMDAT/)(SDY.*)', v['path']).group(2))
            output_image_dir = os.path.join(patient_dir, sub_dir)
            logger.info(
                "processing patient subdirectory {0}".format(output_image_dir))
            if not os.path.exists(output_image_dir):
                os.makedirs(output_image_dir)
            for img, mask in zip(v['imgs'], v['masks']):
                logger.debug(
                    "processing: image -> [{0}] and mask -> {1}".format(img, mask))
                save(img, output_image_dir)

                if mask is not None and os.path.isfile(mask):
                    save(mask, output_image_dir)

            logger.info(
                "complete processing subdirectory {0}".format(output_image_dir))

        data = dict()
        data['patient_id'] = patient_id
        data['age'] = value['age']
        data['sex'] = value['sex']
        with open(os.path.join(patient_dir, 'data.json'), 'w') as fp:
            json.dump(data, fp)
