import argparse
import json
import os
import random

import cv2

from utilities.image_processing import (
    alpha_blend,
    get_contours,
    get_dilated_mask,
    get_mask_contours,
    get_mask_from_contours,
    odd,
    seamless_clone,
    smooth_mask,
)
from utilities.parsing_vaildator import dir_path, file_path


def empty_callback(value):
    pass


# Ensure and set kernel trackbars values to be odd values
def odd_dilation_callback(value):
    value = odd(value)
    cv2.setTrackbarPos("Dilation length", "Output image preview", value)


def odd_gaussian_blur_callback(value):
    # Ensure the minimum value is 1
    if value < 1:
        value = 1
    value = odd(value)
    cv2.setTrackbarPos("Gaussian blur kernel (smooth mask)", "Output image preview", value)


def odd_blurred_mask_callback(value):
    if value < 1:
        value = 1
    value = odd(value)
    cv2.setTrackbarPos("Blur length", "Output image preview", value)


def main(
    coco_filepath: file_path,
    image_library_path: dir_path,
    input_image_name: str,
    output_image_filepath: file_path,
    x_position: int,
    y_position: int,
    scale: float,
    object_index: int,
    method: str,
):
    # Constants
    POSITION_INCREMENT_DECREMENT = 10
    SCALE_INCREMENT_DECREMENT = 0.1
    ALPHA_BLEND_OVERLAY = "alpha_blend"
    SEAMLESS_CLONE_OVERLAY = "seamless_clone"

    # Print input data
    print("--- Synthetic data generator ---")
    print(f"Input image: {os.path.join(image_library_path, input_image_name)}")
    print(f"Output image: {output_image_filepath}")

    # Get data
    image_path = os.path.join(image_library_path, input_image_name)
    image = cv2.imread(image_path)
    cv2.namedWindow("Output image preview")
    # Create trackbars to set parameters
    # Get dilated mask function trackbar
    cv2.createTrackbar("Dilation length", "Output image preview", 51, 255, odd_dilation_callback)
    # Smooth mask function trackbars
    cv2.createTrackbar(
        "Gaussian blur kernel (smooth mask)", "Output image preview", 75, 555, odd_gaussian_blur_callback
    )
    cv2.createTrackbar("Threshold value", "Output image preview", 128, 255, empty_callback)
    cv2.createTrackbar("Max value with thresh binary", "Output image preview", 255, 255, empty_callback)
    # Blurred mask
    cv2.createTrackbar("Blur length", "Output image preview", 149, 255, odd_blurred_mask_callback)
    annotations = json.load(open(coco_filepath))

    # Main loop
    while True:
        # Process mask
        contours, number_of_objects_in_the_photo = get_contours(annotations, input_image_name, object_index)
        mask = get_mask_from_contours(image, contours)

        if mask is not None:
            # Get dilated mask. Bigger dilation_length - more expanded boundaries around litter image
            dilation_length = cv2.getTrackbarPos("Dilation length", "Output image preview")
            dilated = get_dilated_mask(mask, dilation_length)
            # Get smoothed mask. Smaller values of gaussian_blur_kernel will preserve more details, while larger
            # values will result in more smoothing
            gaussian_blur_kernel = cv2.getTrackbarPos("Gaussian blur kernel (smooth mask)", "Output image preview")
            # Pixels with intensity values below threshold_value will be set to 0
            threshold_value = cv2.getTrackbarPos("Threshold value", "Output image preview")
            # For binary thresholding, pixels above the threshold get set to maxvalue
            maxvalue = cv2.getTrackbarPos("Max value with thresh binary", "Output image preview")
            mask_smooth = smooth_mask(dilated, gaussian_blur_kernel, threshold_value, maxvalue)
            # The size of the kernel for the Gaussian blur applied to smoothed mask
            blur_length = cv2.getTrackbarPos("Blur length", "Output image preview")
            mask_blurred = cv2.GaussianBlur(mask_smooth, (blur_length, blur_length), 0)
            mask_blurred = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR)

            mask_contour = get_mask_contours(mask_blurred)
            x, y, w, h = cv2.boundingRect(mask_contour)

            img_cropped = image[y : y + h, x : x + w]
            mask_cropped = mask_blurred[y : y + h, x : x + w]

            if scale is not None:
                if not img_cropped.size == 0:
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

            if method == SEAMLESS_CLONE_OVERLAY:
                output_image_copy[y_offset : y_offset + h, x_offset : x_offset + w] = seamless_clone(
                    output_image_copy[y_offset : y_offset + h, x_offset : x_offset + w], img_cropped, mask_cropped
                )
            elif method == ALPHA_BLEND_OVERLAY:
                output_image_copy[y_offset : y_offset + h, x_offset : x_offset + w] = alpha_blend(
                    output_image_copy[y_offset : y_offset + h, x_offset : x_offset + w], img_cropped, mask_cropped
                )
            else:
                print("Invalid method chosen. Please choose between 'seamless_clone' and 'alpha_blend'.")

            output = output_image_copy

            # Show result
            resized_img_to_show = cv2.resize(output, (1280, 720))
            # resized_img_to_show = cv2.resize(mask_cropped, (1280, 720))
            cv2.imshow("Output image preview", resized_img_to_show)

            # Key control
            key = cv2.waitKey(1) & 0xFF
            if key == ord("w"):
                # Move object up
                y_position -= POSITION_INCREMENT_DECREMENT
            elif key == ord("s"):
                # Move object down
                y_position += POSITION_INCREMENT_DECREMENT
            elif key == ord("a"):
                # Move object left
                x_position -= POSITION_INCREMENT_DECREMENT
            elif key == ord("d"):
                # Move object right
                x_position += POSITION_INCREMENT_DECREMENT
            elif key == ord("z"):
                # Change object
                object_index += 1
                if object_index > number_of_objects_in_the_photo - 1:
                    object_index = 0
                print(
                    f"Number of object in the picture: {number_of_objects_in_the_photo}. "
                    f"Selected item index: {object_index}"
                )
            elif key == ord("x"):
                # Scale object up
                if scale is None:
                    scale = 1.0
                else:
                    scale += SCALE_INCREMENT_DECREMENT
            elif key == ord("c"):
                # Scale object down
                if scale is None:
                    scale = 1.0  # Standard value
                else:
                    scale -= SCALE_INCREMENT_DECREMENT
                if scale < 0.1:
                    scale = SCALE_INCREMENT_DECREMENT
                    print("Minimum scale reached.")
            elif key == ord("e"):
                # Save photo
                cv2.imwrite(output_image_filepath, output)
                print("Photo saved.")
            elif key == ord("q"):
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
                if not img_cropped.size == 0:
                    mask_scale_check = cv2.resize(mask_cropped, (0, 0), fx=scale, fy=scale)
                    h_check, w_check, _ = mask_scale_check.shape
                    while (h_check > bg_h) or (w_check > bg_w):
                        scale -= 0.1
                        mask_scale_check = cv2.resize(mask_cropped, (0, 0), fx=scale, fy=scale)
                        h_check, w_check, _ = mask_scale_check.shape
                        print("Maximum scale reached.")

    print("--- -------------------- ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic data generator")

    parser.add_argument("--coco", type=file_path, help="Pass the COCO file.", required=True)
    parser.add_argument("--library", type=dir_path, help="Photo library connected to COCO file.", required=True)
    parser.add_argument(
        "--input", type=str, help="The name of the photo from which data will be acquired.", required=True
    )
    parser.add_argument(
        "--output", type=file_path, help="Path to photo to which the data will be pasted.", required=True
    )
    parser.add_argument("--position_x", type=int, help="X position of paste.", default=None)
    parser.add_argument("--position_y", type=int, help="Y position of paste.", default=None)
    parser.add_argument("--scale", type=float, help="Scale of paste.", default=None)
    parser.add_argument("--object", type=int, help="Object index on photo.", default=0)
    parser.add_argument(
        "--method",
        type=str,
        choices=["seamless_clone", "alpha_blend"],
        help="Choose the method for blending seamless_clone/alpha_blend.",
        default="seamless_clone",
    )
    args = parser.parse_args()

    main(
        coco_filepath=args.coco,
        image_library_path=args.library,
        input_image_name=args.input,
        output_image_filepath=args.output,
        x_position=args.position_x,
        y_position=args.position_y,
        scale=args.scale,
        object_index=args.object,
        method=args.method,
    )
