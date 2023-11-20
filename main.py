import os
import json
import random
import argparse

import cv2

from utilities.parsing_vaildator import file_path, dir_path
from utilities.image_processing import get_contours, get_mask_from_contours
from utilities.image_processing import process_blurred_mask, get_mask_contours
from utilities.image_processing import alpha_blend


def main(coco_filepath: file_path, image_library_path: dir_path, input_image_name: str,
         output_image_filepath: file_path, x_position: int, y_position: int, scale: float, object_index: int):
    # Print input data
    print(f"--- Synthetic data generator ---")
    print(f"Input image: {os.path.join(image_library_path, input_image_name)}")
    print(f"Output image: {output_image_filepath}")

    # Get data
    image_path = os.path.join(image_library_path, input_image_name)
    image = cv2.imread(image_path)
    annotations = json.load(open(coco_filepath))

    # Process mask
    contours = get_contours(annotations, input_image_name, object_index)
    mask = get_mask_from_contours(image, contours)

    if mask is not None:
        mask_blurred = process_blurred_mask(mask)
        mask_contour = get_mask_contours(mask_blurred)
        x, y, w, h = cv2.boundingRect(mask_contour)

        img_cropped = image[y:y + h, x:x + w]
        mask_cropped = mask_blurred[y:y + h, x:x + w]

        if scale is not None:
            img_cropped = cv2.resize(img_cropped, (0, 0), fx=scale, fy=scale)
            mask_cropped = cv2.resize(mask_cropped, (0, 0), fx=scale, fy=scale)
            h, w, _ = img_cropped.shape

        # Process output image
        output_image = cv2.imread(output_image_filepath)
        bg_h, bg_w, _ = output_image.shape

        if x_position is None:
            x_offset = random.randint(0, bg_w - w)
        else:
            x_offset = x_position

        if y_position is None:
            y_offset = random.randint(0, bg_h - h)
        else:
            y_offset = y_position

        output_image_copy = output_image.copy()

        output_image_copy[y_offset:y_offset + h, x_offset:x_offset + w] = alpha_blend(
            output_image_copy[y_offset:y_offset + h, x_offset:x_offset + w],
            img_cropped, mask_cropped
        )
        output = output_image_copy

        # Show result
        resized_img_to_show = cv2.resize(output, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Output image preview", resized_img_to_show)

        # Save result
        print(f"Press s to save, q to quit.")
        key = cv2.waitKey(0)
        if key == ord('s'):
            cv2.imwrite(output_image_filepath, output)
            print(f"Photo saved.")

    print(f"--- -------------------- ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic data generator')

    parser.add_argument('--coco', type=file_path, help="Pass the COCO file.", required=True)
    parser.add_argument('--library', type=dir_path, help="Photo library connected to COCO file.",
                        required=True)
    parser.add_argument('--input', type=str, help="The name of the photo from which data will be acquired.",
                        required=True)
    parser.add_argument('--output', type=file_path, help="Path to photo to which the data will be pasted.",
                        required=True)
    parser.add_argument('--position_x', type=int, help="X position of paste.", default=None)
    parser.add_argument('--position_y', type=int, help="Y position of paste.", default=None)
    parser.add_argument('--scale', type=float, help="Scale of paste.", default=None)
    parser.add_argument('--object', type=int, help="Object index on photo.", default=0)
    args = parser.parse_args()

    main(coco_filepath=args.coco, image_library_path=args.library, input_image_name=args.input,
         output_image_filepath=args.output, x_position=args.position_x, y_position=args.position_y,
         scale=args.scale, object_index=args.object)
