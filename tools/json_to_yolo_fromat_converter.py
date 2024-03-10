import json
from pathlib import Path
import cv2
import argparse
import sys
utilities_path = Path(__file__).resolve().parent.parent
sys.path.append(str(utilities_path))
from utilities.parsing_vaildator import dir_path, file_path


def convert_json_to_yolo_format(annotation_path: file_path, images_dir_path: dir_path, dest_path: dir_path):
    """
    Converts segmentation mask and bouning box from standard polygon COCO format into YOLO format.
    Args:
        annotation_path: top left corner x coordinate,
        images_dir_path: top left corner y coordinate,
        dest_path: width
    Returns:
        Text files inside bboxes and polygons folders with YOLO format of segmentation masks and bboxes.
    """
    with open(annotation_path, 'r') as f:
        dataset = json.load(f)

    for img_info in dataset['images']:
        img_name = img_info['file_name']
        img_path = Path(images_dir_path) / img_name
        img = cv2.imread(str(img_path))

        img_h, img_w, _ = img.shape

        anns = [ann for ann in dataset['annotations'] if ann['image_id'] == img_info['id']]

        polygons_dest_dir = Path(dest_path) / 'polygons'
        polygons_dest_dir.mkdir(parents=True, exist_ok=True)
        bboxes_dest_dir = Path(dest_path) / 'bboxes'
        bboxes_dest_dir.mkdir(parents=True, exist_ok=True)

        with open(str(polygons_dest_dir / img_name).replace('.jpg', '.txt').replace('.JPG', '.txt').replace('.PNG', '.txt'), 'w') as poly_file, \
             open(str(bboxes_dest_dir / img_name).replace('.jpg', '.txt').replace('.JPG', '.txt').replace('.PNG', '.txt'), 'w') as bbox_file:

            for ann in anns:
                row_poly = '0'
                row_bbox = '0'

                # Check if COCO file is in standard polygon COCO format
                if isinstance(ann['segmentation'], list):
                    # YOLO format from segmentation in the standard format
                    for i in range(0, len(ann['segmentation'][0]), 2):
                        row_poly += f' {ann["segmentation"][0][i] / img_w} {ann["segmentation"][0][i+1] / img_h}'
                    row_poly += '\n'
                    poly_file.write(row_poly)
                else:
                    # Skip annotations with segmentation in the RLE format
                    continue

                # YOLO format from bounding boxes
                bbox = ann['bbox']
                width = bbox[2] / img_w
                height = bbox[3] / img_h

                x_center = (bbox[0] + bbox[2] // 2) / img_w
                y_center = (bbox[1] + bbox[3] // 2) / img_h

                row_bbox += f' {x_center} {y_center} {width} {height}\n'
                bbox_file.write(row_bbox)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert COCO format annotations to YOLO format.')
    parser.add_argument('--coco', type=file_path, help="COCO file that describes objects.", required=True)
    parser.add_argument('--images_dir', type=dir_path, required=True, help='Path to directory containing images.')
    parser.add_argument('--dest_dir', type=dir_path, required=True, help='Path to destination directory for YOLO format annotations.')

    args = parser.parse_args()

    convert_json_to_yolo_format(args.coco, args.images_dir, args.dest_dir)