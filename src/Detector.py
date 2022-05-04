import cv2
import json
import numpy as np
from cv2 import Mat
from typing import List, Tuple
from tqdm import tqdm
from numba import njit
from os import makedirs
from scipy.signal import convolve2d
from Global.utils import load_img_name, getGaussianKernel, checkSlash
from Global.envs import DETECT_BLUR_KSIZE as BKSIZE
from Global.envs import DETECT_BLUR_SIGMA as BSIGMA
from Global.envs import DETECT_GRAD_KSIZE as GKSIZE
from Global.envs import DETECT_SLIDE_KSIZE as SKSIZE
from Global.envs import DETECT_SLIDE_SIGMA as SSIGMA
from Global.envs import DETECT_K as K
from Global.envs import DETECT_R_THRESHOLD as RT
from Global.envs import DETECT_ANMS_NUMBER as N_ANMS
from Global.envs import DETECT_ANMS_RADIUS as R_ANMS
from Global.envs import DRAW_PEN_SIZE as PEN_SIZE

KP_List = List[Tuple[int, int, float]]
pair_int = Tuple[int, int]

@njit
def get_all_local_maxima_points(R, T) -> KP_List:

    kp_list = list()

    # We want to search all local strict maxima and greater than the threshold

    for (row, col), value in np.ndenumerate(R):

        window = R[row-1:row+2, col-1:col+2] # 3x3 window
        if window.shape != (3,3):
            continue

        neighbor = (window - value).flat # neighbor is negative if less than value
        neighbor[4] = -1                 # we hope all neighbors are negative

        allNeg = True
        for n in neighbor:
            isNeg = (n<0)
            allNeg &= isNeg

        if allNeg and value > T:
            kp_list.append((row, col, value))

    kp_list.sort(key=lambda s:s[2])
    return kp_list

def ANMS(height: int, width: int, kp_list: KP_List, radius: int, N_kp:int, border:pair_int) -> List[pair_int]:

    # padding map in order to use kernel
    kp_map = np.zeros((height+2*radius, width+2*radius), dtype=np.bool_)

    # Pixels around border are deprecated
    left, right = border
    kp_map[radius+30:-radius-30, radius+left+30:radius+right-30] = True

    getCircularKenel = lambda r : np.bitwise_not(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1,2*r+1)).astype(np.bool_))
    new_kp_list = list()

    k = radius / 2 / np.log(N_kp+1)

    for i in range(N_kp):

        r = int(radius - k * np.log(i+1))

        # pop until get a valid x and y
        x, y, _ = kp_list.pop()
        while not kp_map[r+x, r+y] and kp_list:
            x, y, _ = kp_list.pop()

        # Mask up the circular range
        kp_map[x:x+2*r+1, y:y+2*r+1] &= getCircularKenel(r)
        new_kp_list.append((x,y))

    return new_kp_list

def HarrisDetector(img: Mat) -> List[pair_int]:

    img = cv2.GaussianBlur(img, (BKSIZE, BKSIZE), BSIGMA, borderType=cv2.BORDER_REFLECT)
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=GKSIZE)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=GKSIZE)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    kernel = getGaussianKernel(SKSIZE, SSIGMA)

    Sx2 = convolve2d(Ix2, kernel, mode='same', boundary='symm')
    Sy2 = convolve2d(Iy2, kernel, mode='same', boundary='symm')
    Sxy = convolve2d(Ixy, kernel, mode='same', boundary='symm')

    R = (Sx2 * Sy2 - Sxy * Sxy) - K * (Sx2 + Sy2) * (Sx2 + Sy2)

    kp_list = get_all_local_maxima_points(R, RT)

    height, width = img.shape
    horizontal_cut = img[img.shape[0]//2, :]
    content = np.nonzero(horizontal_cut)[0]
    left, right = content[0], content[-1]

    kp_list = ANMS(height, width, kp_list, R_ANMS, N_ANMS, (left, right) )

    return kp_list

def draw(img: Mat, kp_list: List[pair_int], out_name: str) -> None:

    for (x, y) in kp_list:
        img = cv2.circle(img, (y,x), PEN_SIZE, (0,0,255), -1)
    cv2.imwrite(out_name, img)

def detect_all(input_dir:str, output_dir:str):

    makedirs(output_dir, exist_ok=True)
    filenames = load_img_name(input_dir)

    with tqdm(total=len(filenames), desc='Detecting: ') as pbar:

        for filename in filenames:

            in_name = input_dir +  filename
            out_name = output_dir + filename
            save_name = out_name.replace(out_name.split('.')[-1], 'json')

            # Corner detection
            im = cv2.imread(in_name)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            kp_list = HarrisDetector(gray)

            # Draw key points
            draw(im, kp_list, out_name)

            # Save key points
            with open(save_name, "w") as fp:
                json.dump(kp_list, fp)

            pbar.update(1)

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    input_dir = checkSlash(args.input_dir)
    output_dir = checkSlash(args.output_dir)

    detect_all(input_dir, output_dir)

