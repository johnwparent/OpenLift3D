import sys, os, distutils
import cv2, numpy as np
import argparse

im_ext = [".png",".jpg"]

def read_in_image(dir):
    return cv2.imread(dir)

def im_to_gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def compute_bin_threshold(im):
    return cv2.threshold(im, 240 , 255, cv2.CHAIN_APPROX_NONE)[1]

def is_rectangle(contour):
    ret = False
    if len(contour) == 4:
        ret = True
    return ret

def find_images_in_dir_rec(dir_):
    files = []
    with os.scandir(dir_) as dirs:
        for direc in dirs:
            if direc.is_file():
                file_type = os.path.splitext(direc.path)[1]
                if file_type in im_ext:
                    files.append(direc.path)
            else:
                find_images_in_dir_rec(direc)
    return files

def create_mask(pts, im):
    im2 = np.zeros_like(im)
    for pt in pts:
        # x,y,w,h = pt
        # cv2.rectangle(im2, (x,y), (x+w,y+h), (255,255,255))
        cv2.drawContours(im, [pt],0,(0,0,0), 5)
    return im2

def apply_mask(im, mask):
    return cv2.bitwise_and(im, mask)

def find_images_in_dirs(dir, recursive=False):
    if recursive:
        return find_images_in_dir_rec(dir)
    else:
        files = []
        with os.scandir(dir) as dirs:
            for direc in dirs:
                if direc.is_file():
                    file_type = os.path.splitext(direc.path)[1]
                if file_type in im_ext:
                    files.append(direc.path)
    return files

def guassian_blur(im):
    return cv2.GaussianBlur(im,(3,3),0)

def derive_countour_data(im):
    img = im_to_gray(im)
    threshold = compute_bin_threshold(img)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours

def find_shape_pts(contour):
    appr = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    if is_rectangle(appr):
        return appr
        # return cv2.boundingRect(appr)
    return None

def drive_masking(im):
    imt = read_in_image(im)
    # imt = guassian_blur(imt)
    contours = derive_countour_data(imt)
    pts_dir = []
    for contour in contours:
        pts = find_shape_pts(contour)
        if pts is not None:
            pts_dir.append(pts)
            cv2.drawContours(imt,[pts],0,(0,0,0),5)

    mask = create_mask(pts_dir, imt)
    masks = cv2.resize(mask, (256, 256))
    imm = apply_mask(imt, mask)
    path, ext = os.path.splitext(im)
    cv2.imwrite(path+"-masked"+ext,imt)


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--file-dir",
        "-fd",
        action="store",
        dest="fd",
        required=True,
    )
    args.add_argument(
        "--recursive",
        "-r",
        action="store",
        dest="recursive",
        required=False,
        default=False,
        type=lambda x: bool(distutils.util.strtobool(x))
    )
    options = args.parse_args(sys.argv[1:])
    im_files = find_images_in_dirs(options.fd,recursive=options.recursive)
    for im in im_files:
        drive_masking(im)
if __name__ == '__main__':
    main()