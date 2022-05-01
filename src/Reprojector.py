import cv2
import numpy as np
from tqdm import tqdm
from numba import njit
from math import floor
from os import listdir, makedirs
from os.path import isdir
from Global.utils import isImage
from Global.envs import REPROJECTION_FOCAL_LENGTH as f
from Global.envs import REPROJECTION_RESIZE_SCALE as s

def resize(img, s):
    height, width, _ = img.shape
    new_h = floor(height * s)
    new_w = floor(width * s)
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return new_img

@njit
def reproject(img, f):

    height, width, _ = img.shape
    half_H = height // 2
    half_W = width // 2

    new_img = np.zeros(img.shape, dtype=np.uint8)

    for row in range(height):
        for col in range(width):

            # Covert the coordinate system
            x = col - half_W
            y = half_H - row

            new_x = f * np.arctan(x/f)
            new_y = f * y / np.sqrt(x**2 + f**2)

            r = round(half_H - new_y)
            c = round(new_x + half_W)

            new_img[r, c, :] = img[row, col, :]

    return new_img

def reproject_dir(input_dir, output_dir):

    if not isdir(input_dir):
        return

    makedirs(output_dir, exist_ok=True)

    filenames = [file for file in listdir(input_dir) if isImage(file)]

    for filename in tqdm(filenames, desc='Reprojecting: '):

        in_name = input_dir + filename
        out_name = output_dir + filename

        im = cv2.imread(in_name)
        im = resize(im, s)
        im = reproject(im, f // 2)
        cv2.imwrite(out_name, im)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    reproject_dir(input_dir, output_dir)
