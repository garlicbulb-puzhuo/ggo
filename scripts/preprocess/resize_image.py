import numpy as np
import cv2
import argparse
from glob import glob
import os


def resize_image(input_dir, output_dir, size):
    file_list = glob(input_dir + "*.npy")
    for f in file_list:
        fname = f.split("/")[-1]
        imgs = np.load(f)
        num = imgs.shape[0]
        resize_imgs = np.ndarray([num, 1, size, size], dtype=np.float32)
        for i in range(imgs.shape[0]):
            resize_imgs[i, 0, :, :] = cv2.resize(
                imgs[i][0], (size, size), interpolation=cv2.INTER_CUBIC)

        np.save(os.path.join(output_dir, fname), resize_imgs)
        print fname + " saved"


def get_parser():
    parser = argparse.ArgumentParser(description='Resize images')
    parser.add_argument('-i', '--input_dir', nargs='?',
                        required=True, help='input path')
    parser.add_argument('-o', '--output_dir', nargs='?',
                        required=True, help='output path')
    parser.add_argument('-s', '--size', nargs='?', type=int,
                        required=True, help='image size')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    resize_image(input_dir=args.input_dir,
                 output_dir=args.output_dir, size=args.size)
