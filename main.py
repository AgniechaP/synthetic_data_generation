import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

from utilities.coco_parser import annotation_poly_based_on_mask, annotation_rle_based_on_mask, get_bbox_based_on_mask
from utilities.image_processing import get_contours, get_mask_from_contours
from utilities.parsing_vaildator import dir_path, file_path
from utilities.pipeline_steps import (
    add_annotation_to_coco,
    add_category_to_coco,
    add_image_to_coco,
    copy_paste_without_blend,
    create_empty_input_coco_file,
    get_background_paths,
    get_objects_from_image,
    get_random_background,
    get_random_image_name_for_object_extraction,
    process_background_image,
    save_output_coco_to_file,
)


def main(
        coco_filepath: file_path,
        image_library_path: dir_path,
        backgrounds_directory_path: dir_path,
        output_direcotry_path: dir_path,
        photo_prefix: str,
        output_photo_number: int,
):
    # Constants
    MIN_NUMBER_OF_OBJECTS_ON_OUTPUT_BACKGROUND = 1
    MAX_NUMBER_OF_OBJECTS_ON_OUTPUT_BACKGROUND = 15
    OUTPUT_COCO_FILE_NAME = "annotations_auto_pipeline.json"
    GENERATED_PHOTO_NAME_SUFFIX = ".jpg"

    print("--- Auto pipeline synthetic data generator ---")

    # Read data from input COCO file
    with open(coco_filepath, "r") as file:
        input_coco_file = json.load(file)

    # Prepare output COCO file
    output_coco_file = create_empty_input_coco_file()
    rubbish_category_id = add_category_to_coco(output_coco_file, "", "rubbish")

    # Check how many background images contains input directory
    background_paths = get_background_paths(backgrounds_directory_path)
    if len(background_paths) == 0:
        print("No background photos were found in input directory.")
        sys.exit()

    # Main loop
    for photo_num in range(output_photo_number):
        # Get random background
        background = get_random_background(background_paths)
        background = process_background_image(background)
        height, width, _ = background.shape

        # Add image record to COCO file
        output_photo_name = photo_prefix + str(photo_num) + GENERATED_PHOTO_NAME_SUFFIX
        coco_photo_id = add_image_to_coco(output_coco_file, width, height, output_photo_name)

        # Initialize the output image with the background and black background as a base for mask
        output_image = background.copy()
        mask_from_generated_photo = np.zeros_like(background, dtype=np.uint8)

        # Initialize a mask to keep track of occupied regions on the background - to avoid overlapping objects
        occupied_mask = np.zeros_like(output_image, dtype=np.uint8)

        # Get random number of objects to add to background
        num_of_objects = random.randint(
            MIN_NUMBER_OF_OBJECTS_ON_OUTPUT_BACKGROUND, MAX_NUMBER_OF_OBJECTS_ON_OUTPUT_BACKGROUND
        )
        for _ in range(num_of_objects):
            # Get random image from which object will be extracted
            object_img_detail = get_random_image_name_for_object_extraction(input_coco_file)
            object_img_path = os.path.join(image_library_path, object_img_detail["file_name"])
            object_img_id = object_img_detail["id"]
            image_with_object = cv2.imread(str(object_img_path))

            # Get segmented objects on image
            segmented_objects_on_image = get_objects_from_image(input_coco_file, object_img_id)

            # Get random object from segmented list
            random_object_index = random.randint(0, len(segmented_objects_on_image) - 1)

            # Get contours of object
            segmentation, _ = get_contours(input_coco_file, object_img_detail["file_name"], random_object_index)

            # Get mask of the rubbish object
            mask = get_mask_from_contours(image_with_object, segmentation)

            # Calculate the scaling factor based on the ratio of original image size to new background size
            scale_factor_x = background.shape[1] / object_img_detail["width"]
            scale_factor_y = background.shape[0] / object_img_detail["height"]
            average_scale_factor = (scale_factor_x + scale_factor_y) / 2

            # Crop the rubbish object and mask
            x, y, w, h = cv2.boundingRect(segmentation)
            rubbish_object = image_with_object[y: y + h, x: x + w]
            rubbish_mask = mask[y: y + h, x: x + w]

            # Determine a scale for the object based on the calculated average scale factor
            scaled_width = int(w * average_scale_factor)
            scaled_height = int(h * average_scale_factor)

            # Resize the rubbish object and mask
            rubbish_object_resized = cv2.resize(rubbish_object, (scaled_width, scaled_height))
            rubbish_mask_resized = cv2.resize(rubbish_mask, (scaled_width, scaled_height))

            # Determine a random position to paste the object onto the background
            paste_x = random.randint(0, background.shape[1] - scaled_width)
            paste_y = random.randint(0, background.shape[0] - scaled_height)

            # Check if the paste area is already occupied
            if np.any(occupied_mask[paste_y: paste_y + scaled_height, paste_x: paste_x + scaled_width]):
                # If occupied, skip this object
                continue

            # Paste the resized rubbish object onto the background at the determined position
            output_image[
                paste_y: paste_y + scaled_height, paste_x: paste_x + scaled_width
            ] = copy_paste_without_blend(
                output_image[paste_y: paste_y + scaled_height, paste_x: paste_x + scaled_width],
                rubbish_object_resized,
                rubbish_mask_resized,
            )

            # Generate mask for each pasted rubbish onto new background
            mask_for_rubbish = np.zeros_like(background, dtype=np.uint8)
            mask_for_rubbish[
                paste_y: paste_y + scaled_height, paste_x: paste_x + scaled_width
            ] = copy_paste_without_blend(
                mask_for_rubbish[paste_y: paste_y + scaled_height, paste_x: paste_x + scaled_width],
                np.ones_like(rubbish_object_resized) * 255,  # White mask for pasted rubbish
                rubbish_mask_resized,
            )

            # Change number of channels to 1
            if mask_for_rubbish.shape[2] != 1:
                mask_for_rubbish = cv2.cvtColor(mask_for_rubbish, cv2.COLOR_BGR2GRAY)
            else:
                pass

            # Generating mask from new image - all rubbish on new background mask
            mask_from_generated_photo[
            paste_y: paste_y + scaled_height, paste_x: paste_x + scaled_width
            ] = copy_paste_without_blend(
                mask_from_generated_photo[paste_y: paste_y + scaled_height, paste_x: paste_x + scaled_width],
                rubbish_object_resized,
                rubbish_mask_resized,
            )

            # Change non-black pixels to white
            mask_from_generated_photo[mask_from_generated_photo != 0] = 255

            # Update the occupied mask with the new object's region
            occupied_mask[paste_y: paste_y + scaled_height, paste_x: paste_x + scaled_width] = 1

            # Create COCO annotation
            poly_annotation = annotation_poly_based_on_mask(mask_for_rubbish)
            if poly_annotation is not None:
                coco_segmentation = poly_annotation[0]
                coco_area = poly_annotation[1]
                coco_iscrowd = 0
            else:
                rle_annotation = annotation_rle_based_on_mask(mask_for_rubbish)
                coco_segmentation = rle_annotation[0]
                coco_area = rle_annotation[1]
                coco_iscrowd = 1

            coco_bbox = get_bbox_based_on_mask(mask_for_rubbish)

            # Add annotation to COCO file
            add_annotation_to_coco(
                output_coco_file,
                coco_photo_id,
                rubbish_category_id,
                coco_segmentation,
                coco_area,
                coco_bbox,
                coco_iscrowd,
            )

        # Save output photo
        output_photo_path = os.path.join(output_direcotry_path, output_photo_name)
        cv2.imwrite(output_photo_path, output_image)

        # Save the mask with all rubish pasted onto new background
        output_mask_path = os.path.join(output_direcotry_path, output_photo_name[:-4] + "_mask.jpg")

        # Change number of channels to 1
        if mask_from_generated_photo.shape[2] != 1:
            mask_from_generated_photo = cv2.cvtColor(mask_from_generated_photo, cv2.COLOR_BGR2GRAY)
        else:
            pass
        cv2.imwrite(output_mask_path, mask_from_generated_photo)

    # Save output COCO file
    save_output_coco_to_file(output_direcotry_path, OUTPUT_COCO_FILE_NAME, output_coco_file)

    print("--- -------------------- ---")


if __name__ == "__main__":
    DEFAULT_OUTPUT_PHOTO_NUMBER = 100

    parser = argparse.ArgumentParser(description="Synthetic data generator")

    parser.add_argument("--coco", type=file_path, help="COCO file that describes objects.", required=True)
    parser.add_argument("--library", type=dir_path, help="Photo library connected to the COCO file.", required=True)
    parser.add_argument("--input", type=dir_path, help="Backgrounds directory for output photos.", required=True)
    parser.add_argument("--output", type=dir_path, help="Output directory to store created files.", required=True)
    parser.add_argument("--prefix", type=str, help="Prefix for generated photos.", required=False, default="agp_photo_")
    parser.add_argument(
        "--number", type=int, help="Number of output files.", required=False, default=DEFAULT_OUTPUT_PHOTO_NUMBER
    )

    args = parser.parse_args()
    main(
        coco_filepath=args.coco,
        image_library_path=args.library,
        backgrounds_directory_path=args.input,
        output_direcotry_path=args.output,
        photo_prefix=args.prefix,
        output_photo_number=args.number,
    )
