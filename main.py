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
    # Constants
    POSITION_INCREMENT_DECREMENT = 10
    SCALE_INCREMENT_DECREMENT = 0.1

    # Print input data
    print(f"--- Synthetic data generator ---")
    print(f"Input image: {os.path.join(image_library_path, input_image_name)}")
    print(f"Output image: {output_image_filepath}")

    # Get data
    image_path = os.path.join(image_library_path, input_image_name)
    image = cv2.imread(image_path)
    annotations = json.load(open(coco_filepath))
    # Main loop
    while True:
        # Process mask
        contours, number_of_objects_in_the_photo = get_contours(annotations, input_image_name, object_index)
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
                x_position = x_offset
            else:
                x_offset = x_position
                if x_position < 0:
                    x_offset = 0
                elif x_position > bg_w - w:
                    x_offset = bg_w - w - 1

            if y_position is None:
                y_offset = random.randint(0, bg_h - h)
                y_position = y_offset
            else:
                y_offset = y_position
                if y_position < 0:
                    y_offset = 0
                elif y_position > bg_h - h:
                    y_offset = bg_h - h - 1

            output_image_copy = output_image.copy()

            output_image_copy[y_offset:y_offset + h, x_offset:x_offset + w] = alpha_blend(
                output_image_copy[y_offset:y_offset + h, x_offset:x_offset + w],
                img_cropped, mask_cropped
            )
            output = output_image_copy

            # Show result
            resized_img_to_show = cv2.resize(output, (1280, 720))
            cv2.imshow("Output image preview", resized_img_to_show)

            # Key control
            key = cv2.waitKey(0)
            if key == ord('w'):
                # Move object up
                y_position -= POSITION_INCREMENT_DECREMENT
            elif key == ord('s'):
                # Move object down
                y_position += POSITION_INCREMENT_DECREMENT
            elif key == ord('a'):
                # Move object left
                x_position -= POSITION_INCREMENT_DECREMENT
            elif key == ord('d'):
                # Move object right
                x_position += POSITION_INCREMENT_DECREMENT
            elif key == ord('z'):
                # Change object
                object_index += 1
                if object_index > number_of_objects_in_the_photo-1:
                    object_index = 0
                print(f"Number of object in the picture: {number_of_objects_in_the_photo}. "
                      f"Selected item index: {object_index}")
            elif key == ord('x'):
                # Scale object up
                if scale is None:
                    scale = 1.0
                else:
                    scale += SCALE_INCREMENT_DECREMENT
            elif key == ord('c'):
                # Scale object down
                if scale is None:
                    scale = 1.0  # Standard value
                else:
                    scale -= SCALE_INCREMENT_DECREMENT
                if scale < 0.1:
                    scale = SCALE_INCREMENT_DECREMENT
                    print(f"Minimum scale reached.")
            elif key == ord('e'):
                # Save photo
                cv2.imwrite(output_image_filepath, output)
                print(f"Photo saved.")
            elif key == ord('q'):
                # Quit
                break

            # Element position protection
            if 0 > y_position:
                y_position = 0
            elif y_position > bg_h - h:
                y_position = bg_h - h

            if 0 > x_position:
                x_position = 0
            elif x_position > bg_w - w:
                x_position = bg_w - w

            # Element scale size protection
            if scale is not None:
                mask_scale_check = cv2.resize(mask_cropped, (0, 0), fx=scale, fy=scale)
                h_check, w_check, _ = mask_scale_check.shape
                while (h_check > bg_h) or (w_check > bg_w):
                    scale -= 0.1
                    mask_scale_check = cv2.resize(mask_cropped, (0, 0), fx=scale, fy=scale)
                    h_check, w_check, _ = mask_scale_check.shape
                    print("Maximum scale reached.")

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
