import cv2
import numpy as np
from glob import glob
import random
import argparse
from pathlib import Path
import sys
utilities_path = Path(__file__).resolve().parent.parent
sys.path.append(str(utilities_path))
from utilities.parsing_vaildator import dir_path


def visualize_yolo_annotations(image_folder: dir_path, bbox_folder: dir_path, polygon_folder: dir_path):
    """
    Visualizes bounding boxes and polygons on images.
    Args:
        image_folder: Path to the folder containing images.
        bbox_folder: Path to the folder containing bounding box labels.
        polygon_folder: Path to the folder containing polygon labels.
    """
    images = glob(image_folder + '/*.jpg')
    random.shuffle(images)

    idx = 0

    while idx < len(images):
        img_path = images[idx]

        img = cv2.imread(img_path)

        h, w = img.shape[:2]

        bbox_label = img_path.replace(image_folder, bbox_folder).replace('.jpg', '.txt')
        polygon_label = img_path.replace(image_folder, polygon_folder).replace('.jpg', '.txt')

        with open(bbox_label, 'r') as f:
            bbox_label = f.readlines()

        with open(polygon_label, 'r') as f:
            polygon_label = f.readlines()

        for row in bbox_label:
            row = row.split(' ')

            x = float(row[1]) * w
            y = float(row[2]) * h

            width = float(row[3]) * w
            height = float(row[4]) * h

            x1 = int(x - width / 2)
            y1 = int(y - height / 2)

            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for row in polygon_label:
            row = row.split(' ')

            points = []

            for i in range(1, len(row), 2):
                x = float(row[i]) * w
                y = float(row[i + 1]) * h

                points.append([x, y])

            points = np.array([points], dtype=np.int32)

            cv2.polylines(img, [points], True, (0, 0, 255), 2)

        # Resize the image window
        resized_img = cv2.resize(img, (1000, 800))  # Set your desired width and height

        cv2.imshow('img', resized_img)

        key = cv2.waitKey(0)

        if key == 27:  # Esc key
            break
        elif key == ord('n'):  # 'n' key
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize bounding boxes and polygons on images.')
    parser.add_argument('--image_folder', type=dir_path, help='Path to the image folder')
    parser.add_argument('--bbox_folder', type=dir_path, help='Path to the folder containing bounding box labels')
    parser.add_argument('--polygon_folder', type=dir_path, help='Path to the folder containing polygon labels')
    args = parser.parse_args()

    visualize_yolo_annotations(args.image_folder, args.bbox_folder, args.polygon_folder)
