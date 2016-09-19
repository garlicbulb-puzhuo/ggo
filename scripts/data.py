#!/usr/bin/env python2.7

import argparse
import os
import logging
import sys
import shutil


logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging_handler_out = logging.StreamHandler(sys.stdout)
logger.addHandler(logging_handler_out)


def parse_options():
    parser = argparse.ArgumentParser(description='Process DICOM Images.')
    parser.add_argument('--input_dir', metavar='input_dir', nargs='?',
                        help='input directory for source images')
    parser.add_argument('--mask_input_dir', metavar='mask_input_dir', nargs='?',
                        help='input directory for mask images')
    parser.add_argument('--output_dir', metavar='output_dir', nargs='?',
                        help='output directory')

    return parser.parse_args()


def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
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
            file_paths.append(os.path.join(root, filename))  # Add it to the list.

    return file_paths  # Self-explanatory.


if __name__ == '__main__':
    args = parse_options()
    # print args.input_dir
    image_path = os.path.abspath(os.path.join(args.input_dir, 'DICOMDAT'))
    mask_path = os.path.abspath(args.mask_input_dir)
    file_paths = get_filepaths(image_path)

    logger.info("root path {0}".format(image_path))

    output_path = os.path.abspath(args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_path in file_paths:
        logger.debug("file path {0}".format(file_path))

        common_prefix = os.path.commonprefix([image_path, file_path])

        relative_path = os.path.relpath(file_path, common_prefix)
        logger.debug("relative path {0}".format(relative_path))

        # check and create the directory if not exists
        parent_dir = os.path.abspath(os.path.join(file_path, os.pardir))
        parent_relative_path = os.path.relpath(parent_dir, common_prefix)
        output_image_dir = os.path.join(output_path, parent_relative_path)
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        mask_filepath = os.path.join(mask_path, "%s-mask.tif" % relative_path)
        logger.debug("mask file path {0}".format(mask_filepath))

        # copy the original image file to the output folder
        shutil.copy2(file_path, output_image_dir)

        if os.path.isfile(mask_filepath):
            # copy the mask image file to the output folder
            logger.info("found matching mask file {0} for {1}".format(mask_filepath, file_path))
            shutil.copy2(mask_filepath, output_image_dir)
        else:
            logger.debug("no matching mask file {0} found for {0}".format(mask_filepath, file_path))


    # print filepaths




