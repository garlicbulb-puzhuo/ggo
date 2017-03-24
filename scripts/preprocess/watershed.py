"""
Watershed pre-processing.
.. https://www.kaggle.com/ankasor/data-science-bowl-2017/improved-lung-segmentation-using-watershed
"""
# Required Imports and loading up a scan for processing as presented by
# Guide Zuidhof

import argparse
import numpy as np  # linear algebra
import dicom
import os
import scipy.ndimage as ndimage
import cv2

from skimage import measure, morphology, segmentation


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[
                                 2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(
            slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


# Some of the starting Code is taken from ArnavJain, since it's more
# readable then my own
def generate_markers(image):
    # Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed


def seperate_lungs(image):
    # Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(
        image)

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    # Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3, 3))
    outline = outline.astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    # Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    # Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    # Use the internal marker and the Outline that was just created to
    # generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # Close holes in the lungfilter
    # fill_holes is not used here, since in some slices the heart would be
    # reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(
        lungfilter, structure=np.ones((5, 5)), iterations=3)

    # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones((512, 512)))

    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed


def get_parser():
    parser = argparse.ArgumentParser(description='watershed data')
    parser.add_argument('-i', '--input-dir', nargs='?', required=True, help='input directory')
    parser.add_argument('-o', '--output-path', nargs='?', required=True, help='output path directory')
    parser.add_argument('-s', '--separate-lung', action='store_true', help='separate lungs')
    parser.add_argument('-r', '--resize', nargs='?', help='resize output images')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    input_dir = args.input_dir
    output_path = args.output_path

    image_rows = 512
    image_cols = 512

    if args.resize is not None:
        image_rows = int(args.resize)
        image_cols = int(args.resize)

    scans = [dir_path for dir_path, _, _ in os.walk(input_dir)][1:]

    for scan in scans:
        p = scan.replace(input_dir, "")
        print "processing " + p
        patient_scan = load_scan(scan)
        imgs = get_pixels_hu(patient_scan)
        num_images = imgs.shape[0]

        out_imgs = np.ndarray(
            [num_images, 1, image_rows, image_cols], dtype=np.float32)
        i = 0
        for img in imgs:
            if i % 10 == 0:
                print 'image ' + str(i)

            if args.seperate_lung:
                segmented, _, _, _, _, _, _, _ = seperate_lungs(img)
                intermediate_img = segmented
            else:
                intermediate_img = img

            if args.resize is not None:
                intermediate_img = cv2.resize(
                    intermediate_img, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)

            out_imgs[i, 0, :, :] = intermediate_img
            i += 1

        np.save(output_path + p + ".npy", out_imgs)
