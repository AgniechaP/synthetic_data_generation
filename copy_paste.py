import cv2
import numpy as np
import os
import json
import math
import random

# TODO: argument parser of paths to images, image filename, annotations, background
# TODO: optionally add possibility to choose background image positions to paste blended litter
# TODO: functionality of saving generated images with pasted litter in some directory

# Set of data to be replaced with argument parser
annotations_path = '/home/agnieszka/Documents/UAVVaste/annotations/annotations.json'
images_directory = '/home/agnieszka/Documents/UAVVaste/images/'
image_filename = 'batch_05_img_1780.jpg'
background_directory = '/home/agnieszka/Documents/UAVVaste/images/batch_01_frame_12.jpg'


def get_contours(images_dir, image_name, annotations):
    """
    Returns segmentation points

    Arguments:
    images_dir  – path to directory with set of images
    image_name  - image on which litter is labeled
    annotations – json file with coco annotations
    """
    path_to_image = os.path.join(images_dir, image_name)
    annotation = next((anno for anno in annotations['annotations'] if os.path.join(images_dir, annotations['images'][anno['image_id']]['file_name']) == path_to_image),None)
    if annotation:
        segmentation = annotation['segmentation'][0]
        segmentation = np.array(segmentation, np.int32)
        segmentation = segmentation.reshape((-1, 1, 2))
    else:
        print(f"No annotations found on image")
        segmentation = None
    return segmentation


def mask_from_contours(image, contours):
    """
    Returns a mask - white trash on black background

    Arguments:
    image    – image on which litter is labeled
    contours – points of segmentation
    """
    mask = np.zeros_like(image, dtype=np.uint8)
    return cv2.fillPoly(mask, [contours], (255, 255, 255))


def dilate_mask(mask, kernel_size=11):
    """
    Returns a dilated mask

    Arguments:
    mask        – masked image
    kernel_size – dimension of the kernel
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated


def threshold(img, thresh=128, maxval=255, type=cv2.THRESH_BINARY):
    """
    Returns a tresholded binary mask

    Arguments:
    img    – masked image in color or grey scale
    thresh – trshold value
    maxval - intensity value assigned to the pixels that meet treshold
    type   - type of thresholding operation to perform
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshed = cv2.threshold(img, thresh, maxval, type)[1]
    return threshed


def smooth_mask(mask, kernel_size=11):
    """
    Returns a blurred mask after GaussianBlur operation
    """
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    threshed = threshold(blurred)
    return threshed


def odd(num):
    """
    Returns an odd integer number
    """
    if isinstance(num, float):
        num = math.floor(num)
    if num % 2 == 0:
        num = num - 1
    return num


def find_contours(blurred_mask):
    """
    Returns a blurred mask

    Arguments:
    blurred_mask - smoothed and blurred mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[-2]


def max_contour(contours):
    return sorted(contours, key=cv2.contourArea)[-1]


def alpha_blend(background, foreground, mask):
    """
    Returns an alpha blending combining a foreground image with a background image based on a mask

    Arguments:
    background - background image
    foreground - foreground image
    mask       - binary mask
    """
    mask = mask.astype("float") / 255.
    foreground = foreground.astype("float") / 255.
    background = background.astype("float") / 255.
    out = background * (1 - mask) + foreground * mask
    out = (out * 255).astype("uint8")
    return out


def main():
    # Connection of directories to get the full image path
    image_path = os.path.join(images_directory, image_filename)

    image = cv2.imread(image_path)
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotations = json.load(open(annotations_path))

    dilation_length = 51
    blur_length = 115

    contours = get_contours(images_directory, image_filename, annotations)
    mask = mask_from_contours(image, contours)
    if mask is not None:
        mask_dilated = dilate_mask(mask, dilation_length)
        mask_smooth = smooth_mask(mask_dilated, odd(dilation_length * 1.5))
        mask_blurred = cv2.GaussianBlur(mask_smooth, (blur_length, blur_length), 0)
        mask_blurred = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR)

        mask_threshed = threshold(mask_blurred, 1)
        mask_contours = find_contours(mask_threshed)
        mask_contour = max_contour(mask_contours)

        x, y, w, h = cv2.boundingRect(mask_contour)

        img_cropped = image[y:y + h, x:x + w]
        mask_cropped = mask_blurred[y:y + h, x:x + w]

        # Background image - to be replaced with the proper environment background image from arg parser
        background = cv2.imread(background_directory)

        # Generating random coordinates on bcg image to do blending trash cropped image + background
        bg_h, bg_w, _ = background.shape
        x_offset = random.randint(0, bg_w - w)
        y_offset = random.randint(0, bg_h - h)

        # Keeping the original image of background to paste blended litter on it
        background_copy = background.copy()

        # Replacing random square on background with alpha blended image (blending image with part of background)
        background_copy[y_offset:y_offset + h, x_offset:x_offset + w] = alpha_blend(background_copy[y_offset:y_offset + h, x_offset:x_offset + w], img_cropped, mask_cropped)
        output = background_copy

        # Previous try according to stackoverflow answer -
        # to view it, replace lines from background = to output = background copy
        # background = np.full(img_cropped.shape, (200, 240, 200), dtype=np.uint8)
        # output = alpha_blend(background, img_cropped, mask_cropped)

        resized_img_to_show = cv2.resize(output, (0, 0), fx=0.3, fy=0.3)
        cv2.imshow('Image', resized_img_to_show)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



