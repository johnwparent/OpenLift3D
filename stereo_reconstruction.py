import os, sys
import cv2
import argparse


def main():
    args.argparse.ArgumentParser()
    args.add_argument(
        "--left",
        required=True,
        dest="left",
        action="store",
    )
    args.add_argument(
        "--right",
        required=True,
        dest="right",
        action="store",
    )
    args.add_argument(
        "--output",
        "-o",
        required=False,
        default=os.getcwd(),
        dest="output",
        action="store",
    )
    opts = args.parse_args()
    left = cv2.imread(opts.left)
    right = cv2.imread(opts.right)

    stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left,right)
    cv2.imwrite("stero_im",opts.output)

if __name__ == '__main__':
    main()