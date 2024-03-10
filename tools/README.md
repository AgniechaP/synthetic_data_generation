# Tools
This folder contains helpful scripts for various operations related to handling annotations and visualizing bounding boxes and masks. 
## Scripts
1. **json_to_yolo_format_converter.py**: 

This script converts a standard polygon COCO JSON file into TXT YOLO format files, storing them inside the 'bboxes' and 'polygons' folders. The following arguments are required:
```bash
python3 json_to_yolo_format_converter.py --coco /path/to/coco/file \ 
--images_dir /path/to/images/directory/ \ 
--dest_dir /path/to/destination/directory/
```
2. **visualize_bbox_from_coco.py**: 

This script displays an image with bounding box annotations from a COCO JSON file. Press random key on keyboard to switch to next bouning box around an object. Press `Esc` to exit the image display. The arguments required are:

```bash
python3 visualize_bbox_from_coco.py \ 
--image_folder /path/to/image/folder/ \
--annotation_file /path/to/annotation/file \
--window_width 800 --window_height 600
```
3. **visualize_mask_and_bbox_from_yolo.py**: 

This script visualizes segmentation masks and bounding boxes saved in YOLO format TXT files. It reads files from the 'bboxes' and 'polygons' folders. Press `n` to switch to another image with visualized bouning boxes and polygons. Press `Esc` to exit the image display.

```bash
python3 visualize_mask_and_bbox_from_yolo.py \ 
--image_folder /path/to/image/folder/ \
--bbox_folder /path/to/bboxes/folder/ \
--polygon_folder /path/to/polygons/folder/
```