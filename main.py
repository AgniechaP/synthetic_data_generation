import argparse

from utilities.parsing_vaildator import *


def main(coco_filepath: file_path, image_library_path: dir_path, input_image_name: str,
         output_image_filepath: file_path, x_position: int, y_position: int, scale: float):

    pass


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
    args = parser.parse_args()

    main(coco_filepath=args.coco, image_library_path=args.library, input_image_name=args.input,
         output_image_filepath=args.output, x_position=args.position_x, y_position=args.position_y,
         scale=args.scale)
