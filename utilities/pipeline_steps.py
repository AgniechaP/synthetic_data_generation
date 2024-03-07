import datetime
import json
import os
import random

import cv2
import numpy
import numpy as np

from utilities.parsing_vaildator import dir_path


def create_empty_input_coco_file() -> dict:
    """
    Creates empty coco file.
    Returns:
    Dictionary in COCO format.
    """
    output_coco_file = {
        "images": [],
        "categories": [],
        "annotations": [],
        "licenses": [],
        "info": {
            "year": datetime.datetime.now().year,
            "version": "v1",
            "description": "Auto generated data from pipeline.",
            "contributor": "None",
            "url": "https://github.com/AgniechaP/synthetic_data_generation",
            "date_created": str(datetime.datetime.now()),
        },
    }
    return output_coco_file


def add_category_to_coco(coco_file: dict, supercategory: str, name: str) -> int:
    """
    Add category to "category" section in COCO.
    Args:
        coco_file: Coco file to which data will be added.
        supercategory: Supercategory name.
        name: Category name.
    Returns:
    Last id of category added.
    """
    if len(coco_file["categories"]) != 0:
        last_category_id = coco_file["categories"][-1]["id"]
    else:
        last_category_id = -1

    current_category_id = last_category_id + 1
    category_body = {"supercategory": supercategory, "id": current_category_id, "name": name}

    coco_file["categories"].append(category_body)
    return current_category_id


def add_image_to_coco(coco_file: dict, width: int, height: int, file_name: str) -> int:
    """
    Add image record to "images" list in COCO file.
    Args:
        coco_file: Coco file to which data will be added.
        width: Photo width.
        height: Photo height.
        file_name: Name of the photo.
    Returns:
    Last id of photo added.
    """
    if len(coco_file["images"]) != 0:
        last_image_id = coco_file["images"][-1]["id"]
    else:
        last_image_id = -1

    current_image_id = last_image_id + 1
    image_body = {
        "id": current_image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
        "license": None,
        "flickr_url": None,
        "coco_url": None,
        "date_captured": None,
        "flickr_640_url": None,
    }
    coco_file["images"].append(image_body)

    return current_image_id


def add_annotation_to_coco(
    coco_file: dict, image_id: int, category_id: int, segmentation: list[list], area: float, bbox: list, iscrowd: int
) -> int:
    """
    Add annotation record to "annotations" list in COCO file.
    Args:
        coco_file: Coco file to which data will be added.
        image_id: Image id.
        category_id: Category of annotation.
        segmentation: List of lists of segmentation data.
        area: Area of segmentation.
        bbox: BoundingBox.
        iscrowd: Is annotation crowd.
    Returns:
    Last id of annotation added.
    """
    if len(coco_file["annotations"]) != 0:
        last_annotation_id = coco_file["annotations"][-1]["id"]
    else:
        last_annotation_id = -1

    current_annotation_id = last_annotation_id + 1
    annotation_body = {
        "id": current_annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": area,
        "bbox": bbox,
        "iscrowd": iscrowd,
    }
    coco_file["annotations"].append(annotation_body)

    return current_annotation_id


def get_background_paths(backgrounds_directory_path: dir_path) -> list:
    """
    Process and validate path to directory containing background images and get list of background paths.
    Args:
        backgrounds_directory_path: path to directory containing background images.
    Returns:
        List of background paths.
    """
    background_paths = []

    files_in_directory = os.listdir(backgrounds_directory_path)
    for file in files_in_directory:
        background_path = os.path.join(backgrounds_directory_path, file)
        try:
            photo = cv2.imread(background_path)
        except:
            continue
        if photo is not None:
            background_paths.append(background_path)

    return background_paths


def get_random_background(background_paths: list) -> numpy.ndarray:
    """
    Get random background from input paths.
    Args:
        background_paths: List of paths to background photos.
    Returns:
        Image in RGB OpenCV format.
    """
    background_index = random.randint(0, len(background_paths) - 1)
    background_path = background_paths[background_index]
    # background = cv2.cvtColor(cv2.imread(background_path), cv2.COLOR_BGR2RGB)
    background = cv2.imread(background_path)

    return background


def process_background_image(background: numpy.ndarray):
    """
    Processing background image.
    Args:
        background: Background image in RGB OpenCV format.
    Returns: Processed image in RGB OpenCV format.
    """
    # Optional: implement needed processing.
    return background


def get_random_image_name_for_object_extraction(coco_dict: dict) -> dict:
    """

    Args:
        coco_dict:
    Returns:

    """
    number_of_available_images = len(coco_dict["images"])
    image_index = random.randint(0, number_of_available_images - 1)
    return coco_dict["images"][image_index]


def get_objects_from_image(coco_dict: dict, image_id: int) -> list:
    """

    Args:
        coco_dict:
        image_id:

    Returns:

    """
    segmented_objects = []
    for annotation in coco_dict["annotations"]:
        if annotation["image_id"] == image_id:
            segmented_objects.append(annotation)
    return segmented_objects


def save_output_coco_to_file(output_directory: dir_path, file_name: str, coco_dictionary: dict):
    """
    Saves input dictionary to JSON in given file.
    Args:
        output_directory: Path to output dictionary.
        file_name: Name of output COCO file.
        coco_dictionary: Dictionary with COCO data.
    """
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as file:
        json.dump(coco_dictionary, file, indent=4)


def copy_paste_without_blend(background: numpy.ndarray, foreground: numpy.ndarray, mask: numpy.ndarray) -> np.ndarray:
    """
    Paste foreground (rubbish) onto the background based on a mask without blending.
    Args:
        background: Background image
        foreground: Foreground image.
        mask: Binary mask.
    Returns:
        Output image.
    """
    # Ensure that background and foreground have the same shape
    if background.shape != foreground.shape:
        foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

    # Check if the mask is not None and not empty
    if (mask is None) or (mask.size == 0):
        print("Warning: Empty mask. Unable to paste foreground onto background.")
        return background  # Return the original background if the mask is empty

    # Check if the mask has a valid size
    if (mask.shape[0] <= 0) or (mask.shape[1] <= 0):
        print("Warning: Invalid mask size. Unable to paste foreground onto background.")
        return background

    # Resize mask to match the shape of the background
    mask_resized = cv2.resize(mask, (background.shape[1], background.shape[0]))

    # Convert mask to binary
    mask_resized = (mask_resized > 128).astype(numpy.uint8)

    # Copy pixels from foreground (rubbish) to background using the mask
    output_image = background.copy()
    output_image[mask_resized != 0] = foreground[mask_resized != 0]

    return output_image
