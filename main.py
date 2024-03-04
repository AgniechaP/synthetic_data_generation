import argparse
import json
import os
import random
import sys

import cv2

from utilities.parsing_vaildator import dir_path, file_path
from utilities.pipeline_steps import (
    add_category_to_coco,
    add_image_to_coco,
    create_empty_input_coco_file,
    get_background_paths,
    get_random_background,
    get_random_image_name_for_object_extraction,
    process_background_image,
    save_output_coco_to_file,
    get_objects_from_image
)

from utilities.image_processing import get_contours


def main(
    coco_filepath: file_path,
    image_library_path: dir_path,
    backgrounds_directory_path: dir_path,
    output_direcotry_path: dir_path,
    output_photo_number: int,
):
    # Constants
    MIN_NUMBER_OF_OBJECTS_ON_OUTPUT_BACKGROUND = 1
    MAX_NUMBER_OF_OBJECTS_ON_OUTPUT_BACKGROUND = 15
    OUTPUT_COCO_FILE_NAME = "annotations_auto_pipeline.json"
    GENERATED_PHOTO_NAME_PREFIX = "agp_photo_"
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
        output_photo_name = GENERATED_PHOTO_NAME_PREFIX + str(photo_num) + GENERATED_PHOTO_NAME_SUFFIX
        coco_photo_id = add_image_to_coco(output_coco_file, width, height, output_photo_name)

        # Get random number of objects to add to background
        num_of_objects = random.randint(
            MIN_NUMBER_OF_OBJECTS_ON_OUTPUT_BACKGROUND, MAX_NUMBER_OF_OBJECTS_ON_OUTPUT_BACKGROUND
        )
        for _ in range(num_of_objects):
            # Get random image from which object will be extracted
            object_img_detail = get_random_image_name_for_object_extraction(input_coco_file)
            object_img_path = os.path.join(image_library_path, object_img_detail["file_name"])
            object_img_id = object_img_detail["id"]

            # Get segmented objects on image
            segmented_objects_on_image = get_objects_from_image(input_coco_file, object_img_id)

            # Get random object from segmented list
            random_object_index = random.randint(0, len(segmented_objects_on_image)-1)
            random_object = segmented_objects_on_image[random_object_index]

            # Calculate width and height ratio
            x_w = width * random_object["bbox"][2] / object_img_detail["width"]
            x_h = height * random_object["bbox"][3] / object_img_detail["height"]

            # Get contours and mask of object


            # 6. Wklej obiekt na tło - maska,
            # 7. Przygotuj dane do pliku COCO na podstawie maski,
            # 8. Zapisz annotation do pliku COCO,

        # Save output photo
        output_photo_path = os.path.join(output_direcotry_path, output_photo_name)
        # cv2.imwrite(output_photo_path, img)  # TODO: replace img with proper image name

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
    parser.add_argument(
        "--number", type=int, help="Number of output files.", required=False, default=DEFAULT_OUTPUT_PHOTO_NUMBER
    )

    args = parser.parse_args()
    main(
        coco_filepath=args.coco,
        image_library_path=args.library,
        backgrounds_directory_path=args.input,
        output_direcotry_path=args.output,
        output_photo_number=args.number,
    )
