"""Script to exeperiment different images formats"""

import argparse
import sys

from image import look_transformation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("image", type=str)
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)

    args = parser.parse_args(sys.argv[1:])

    look_transformation(args.image, args.width, args.height)
