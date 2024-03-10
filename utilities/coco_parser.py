from typing import Dict, List, Tuple

import cv2
import numpy as np


def annotation_rle_based_on_mask(mask: np.ndarray) -> Tuple[Dict, int]:
    """
    Creates annotation for COCO format (RLE) based on mask.
    Args:
        mask: Mask of an object (represented by 255) and background (represented by 0).
    Returns:
    Segmentation section of COCO file in iscrowd: 1 format (RLE) and area.
    """
    height, width = mask.shape
    size = [height, width]
    counts = []

    area = 0
    bcg_count = 0
    obj_count = 0
    last_value = 0
    for h in range(height):
        for w in range(width):
            if mask[h, w] > 0:
                if last_value == 0:
                    counts.append(bcg_count)
                    bcg_count = 0
                area += 1
                obj_count += 1
                last_value = 1
            else:
                if last_value > 0:
                    counts.append(obj_count)
                    obj_count = 0
                bcg_count += 1
                last_value = 0

    if obj_count > 0:
        counts.append(obj_count)
    if bcg_count > 0:
        counts.append(bcg_count)

    segmentation = {"size": size, "counts": counts}
    return segmentation, area


def get_bbox_based_on_mask(mask: np.ndarray):
    """
    Calculate bbox of object based on mask (COCO format).
    - x, y: the upper-left coordinates of the bounding box,
    - width, height: the dimensions of your bounding box.
    Args:
        mask: Mask of an object (represented by 255) and background (represented by 0).
    Returns:
    [x, y, w, h] coordinates starting from left top corner.
    """
    object_points = np.where(mask > 0)
    x = int(np.min(object_points[1]))
    y = int(np.min(object_points[0]))
    w = int(np.max(object_points[1]) - x + 1)
    h = int(np.max(object_points[0]) - y + 1)
    return [x, y, w, h]


def convert_coco_bbox_to_yolo_format(x, y, w, h):
    """
    Converts top left x and y coordinates convention of bbox to center convention.
    Args:
        x: top left corner x coordinate,
        y: top left corner y coordinate,
        w: width,
        h: height.
    Returns:
    [x, y, w, h] coordinates starting from center.
    """
    x = x + w // 2
    y = y + h // 2
    return [x, y, w, h]


def annotation_poly_based_on_mask(mask: np.ndarray) -> Tuple[List, float] | None:
    """
    Creates annotation for COCO format (polygons) based on mask.
    Args:
        mask: Mask of an object (represented by 255) and background (represented by 0).
    Returns:
        Segmentation section of COCO file in iscrowd: 0 format (polygons) and area.
    """
    try:
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        print("No contours found")
        return None

    segmentation_parts = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation_parts.append(contour)

    if len(segmentation_parts) == 0:
        print("Contours on mask were not found!")
        return None

    area = len(np.where(mask > 0)[0])
    return segmentation_parts, area


def calculate_area_of_poly_annotation(x: list, y: list) -> float:
    """
    Calculate area of annotation based on points
    Args:
        x: List of x coordinates.
        y: List of y coordinates.

    Returns:
    Area of segmentation.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
