import math

import cv2
import numpy as np


def get_contours(coco_annotations: dict, input_image_name: str, object_index: int = 0):
    """
    Get segmentation points
    Args:
        coco_annotations: json file with coco annotations
        input_image_name: path to directory with set of images
        object_index: index of object to be cut from original photo
    Returns: (segmentation points, number of object in the photo)
    """
    objects_on_image = []
    image_id = None

    for image in coco_annotations["images"]:
        if image["file_name"] == input_image_name:
            image_id = image["id"]
            break

    if image_id is None:
        print(f"No image with name {input_image_name} found in annotation file!")
        return None

    for annotation in coco_annotations["annotations"]:
        if annotation["image_id"] == image_id:
            objects_on_image.append(annotation["segmentation"])

    if len(objects_on_image) == 0:
        print(f"No objects found on image {input_image_name}!")
        return None

    segmentation = objects_on_image[object_index]
    segmentation = np.array(segmentation, np.int32)
    segmentation = segmentation.reshape((-1, 1, 2))
    number_of_object_in_the_photo = len(objects_on_image)

    return segmentation, number_of_object_in_the_photo


def get_mask_from_contours(image: np.ndarray, contours):
    """
    Get a mask - white trash on black background
    Args:
        image: image on which litter is labeled
        contours: points of segmentation
    Returns: mask of object
    """
    mask = np.zeros_like(image, dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return cv2.fillPoly(mask, [contours], (255, 255, 255))


def get_dilated_mask(mask: np.ndarray, kernel_size: int = 11):
    """
    Get a dilated mask of the picture.
    Args:
        mask: mask of object in image
        kernel_size: dimension of the kernel
    Returns: dilated mask of the input mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)

    return dilated


def threshold(img: np.ndarray, thresh: float = 128, maxvalue: float = 255, dtype: int = cv2.THRESH_BINARY):
    """
    Performs a threshold operation in input image.
    Args:
        img: masked image in color or grey scale
        thresh: threshold value
        maxvalue: intensity value assigned to the pixels that meet threshold
        dtype: type of thresholding operation to perform
    Returns: thresholded binary mask
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshed = cv2.threshold(img, thresh, maxvalue, dtype)[1]

    return threshed


def smooth_mask(mask: np.ndarray, kernel_size: int = 11):
    """
    Smooths input mask by performing a Gaussian blur and threshold.
    Args:
        mask: mask of object in image
        kernel_size: dimension of the kernel
    Returns: blurred mask after GaussianBlur operation
    """
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    threshed = threshold(blurred)

    return threshed


def odd(num):
    """
    Get odd number.
    Args:
        num: number
    Returns: odd integer number
    """
    if isinstance(num, float):
        num = math.floor(num)
    if num % 2 == 0:
        num = num - 1

    return num


def find_contours(blurred_mask: np.ndarray):
    """
    Finds contours on blurred mask
    Args:
        blurred_mask: smoothed and blurred mask
    Returns: contours list
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours[-2]


def get_max_contour(contours):
    """
    Gets maximum contour of the input contours
    Args:
        contours: list of contours
    Returns: maximum contour
    """
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]


def alpha_blend(background: np.ndarray, foreground: np.ndarray, mask: np.ndarray):
    """
    Performs alpha blending combining a foreground image with a background image based on a mask
    Args:
        background: background image
        foreground: foreground image
        mask: binary mask
    Returns: output image
    """
    mask = mask.astype("float") / 255.
    foreground = foreground.astype("float") / 255.
    background = background.astype("float") / 255.
    out = background * (1 - mask) + foreground * mask
    out = (out * 255).astype("uint8")

    return out

def seamless_clone(background: np.ndarray, foreground: np.ndarray, mask: np.ndarray):
    """
    Performs seamless cloning technique combining a foreground image with a background image based on a mask
    Args:
        background: background image
        foreground: foreground image
        mask: binary mask
    Returns: output image
    """

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Finding the center of the mask contour
    center = cv2.boundingRect(mask_gray)[0] + cv2.boundingRect(mask_gray)[2] // 2, \
             cv2.boundingRect(mask_gray)[1] + cv2.boundingRect(mask_gray)[3] // 2

    # Performing seamless cloning
    out = cv2.seamlessClone(foreground, background, mask_gray, center, cv2.NORMAL_CLONE)

    return out


def process_blurred_mask(mask: np.ndarray, dilation_length: int = 51, blur_length: int = 149):
    """
    Processes input mask to be blurred and dilated
    Args:
        mask: mask of the object in the image
        dilation_length: dilatation parameter
        blur_length: blur parameter
    Returns: blurred mask
    """
    mask_dilated = get_dilated_mask(mask, dilation_length)
    mask_smooth = smooth_mask(mask_dilated, odd(dilation_length * 1.5))
    mask_blurred = cv2.GaussianBlur(mask_smooth, (blur_length, blur_length), 0)
    mask_blurred = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR)

    return mask_blurred


def get_mask_contours(mask: np.ndarray):
    """
    Gets mask contour
    Args:
        mask: mask of the object in image
    Returns: maks contour
    """
    mask_threshed = threshold(mask, 1)
    mask_contours = find_contours(mask_threshed)
    mask_contour = get_max_contour(mask_contours)

    return mask_contour
