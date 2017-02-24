import argparse
import os

# sys.path.insert(0, '../luna16_script/')

import numpy as np

from watershed import load_scan, get_pixels_hu, seperate_lungs


def get_parser():
    parser = argparse.ArgumentParser(description='watershed data')
    parser.add_argument('-i', '--input-dir', nargs='?', required=True,
                        help='input directory')
    parser.add_argument('-o', '--output-path', nargs='?', required=True,
                        help='output path directory')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    input_dir = args.input_dir
    output_path = args.output_path

    scans = [dirpath for dirpath, dirnames, files in os.walk(input_dir)][1:]
    for scan in scans:
        p = scan.replace(input_dir, "")
        print "processing " + p
        patient_scan = load_scan(scan)
        imgs = get_pixels_hu(patient_scan)
        num_images = imgs.shape[0]
        out_imgs = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
        i = 0
        for img in imgs:
            if i % 10 == 0:
                print 'imgage ' + str(i)
            segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed = seperate_lungs(
                img)
            out_imgs[i, 0, :, :] = segmented
            i += 1
        np.save(output_path + p + ".npy", out_imgs)
