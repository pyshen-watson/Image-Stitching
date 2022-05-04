import cv2
import numpy as np
from cv2 import Mat
from tqdm import tqdm
from numba import njit
from os import makedirs
from Global.utils import checkSlash, load_img_name
from Global.envs import REPROJECTION_FOCAL_LENGTH as f
from Global.envs import REPROJECTION_RESIZE_SCALE as s

def resize_by_scale(img: Mat, s: float) -> Mat:

    height, width, _ = img.shape
    new_h = int(height * s)
    new_w = int(width * s)
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return new_img

@njit
def reproject(img: Mat, f_len: float) -> Mat:

    height, width, _ = img.shape
    half_H = height // 2
    half_W = width // 2

    new_img = np.zeros(img.shape, dtype=np.uint8)

    for row in range(height):
        for col in range(width):

            # Covert the coordinate system
            # Let the image center be origin point
            x = col - half_W
            y = half_H - row

            new_x = f_len * np.arctan(x/f_len)
            new_y = f_len * y / np.sqrt(x**2 + f_len**2)


            # Covert the coordinate system back
            r = int(half_H - new_y)
            c = int(new_x + half_W)

            new_img[r, c, :] = img[row, col, :]

    return new_img

def reproject_all(input_dir: str, output_dir: str) -> None:

    makedirs(output_dir, exist_ok=True)

    filenames = load_img_name(input_dir)

    with tqdm(total=len(filenames), desc='Reprojecting: ') as pbar:

        for filename in filenames:

            in_name = input_dir + filename
            out_name = output_dir + filename

            im = cv2.imread(in_name)
            im = resize_by_scale(im, s)
            im = reproject(im, f)
            cv2.imwrite(out_name, im)

            pbar.update(1)

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    input_dir = checkSlash(args.input_dir)
    output_dir = checkSlash(args.output_dir)

    reproject_all(input_dir, output_dir)
