import os
import cv2
import json
import sys
from pathlib import Path
import argparse
utilities_path = Path(__file__).resolve().parent.parent
sys.path.append(str(utilities_path))
from utilities.parsing_vaildator import dir_path, file_path


def visualize_bbox(image_folder: dir_path, annotation_file: file_path, window_width: int, window_height: int):
    """
    Visualizes bouning boxes saved in COCO annotation json file on image. 
    Args:
        image_folder: directory with images described inside COCO file,
        annotation_file: file with standard polygon COCO annotations,
        window_width: width of shown image,
        window_height: height of shown image
    Returns:
        Imshow of an image with drawn rectangles around objects.
    """
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Load images
    images = {image['id']: image['file_name'] for image in annotations['images']}
    
    # Display images with bounding boxes
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        image_file = os.path.join(image_folder, images[image_id])
        image = cv2.imread(image_file)
        
        # Get bounding box coordinates
        x, y, width, height = map(int, annotation['bbox'])
        original_height, original_width, _ = image.shape
        scale_x = window_width / original_width
        scale_y = window_height / original_height

        # Scale image and bounding box x, y coordinates
        image = cv2.resize(image, (window_width, window_height))
        x1 = int(x*scale_x)
        y1 = int(y*scale_y)
        x2 = int((x + width) * scale_x)
        y2 = int((y + height) * scale_y)

        # Draw bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Display the image
        cv2.imshow('Image with Bounding Box', image)
        
        # Wait for key press
        key = cv2.waitKey(0)
        
        # Check if the Esc key (key code 27) is pressed
        if key == 27:
            break
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize bounding boxes on images.')
    parser.add_argument('--image_folder', type=dir_path, help='Path to the image folder')
    parser.add_argument('--annotation_file', type=file_path, help='Path to the annotation file')
    parser.add_argument('--window_width', type=int, default=800, help='Width of the displayed window')
    parser.add_argument('--window_height', type=int, default=600, help='Height of the displayed window')
    args = parser.parse_args()

    visualize_bbox(args.image_folder, args.annotation_file, args.window_width, args.window_height)
