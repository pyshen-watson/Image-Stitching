import cv2
import json
import numpy as np
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

@njit
def getKPs(R, T):

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

def ANMS(height, width, kp_list, R, N):

    # padding map in order to use kernel
    kp_map = np.ones((height+2*R, width+2*R), dtype=np.bool_)
    getCircularKenel = lambda r : np.bitwise_not(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1,2*r+1)).astype(np.bool_))


    new_kp_list = list()

    for i in range(N):

        r = R - (R * i) // N

        # pop until get a valid x and y
        while True:
            x, y, value = kp_list.pop()
            if kp_map[r+x, r+y]:
                break

        # Mask up the circular range
        kp_map[x:x+2*r+1, y:y+2*r+1] &= getCircularKenel(r)
        new_kp_list.append((x,y))

    return new_kp_list

def HarrisDetector(img):

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

    height, width = img.shape
    kp_list = getKPs(R, RT)
    kp_list = ANMS(height, width, kp_list, R_ANMS, N_ANMS)

    return kp_list

def detect_all(input_dir, output_dir):

    makedirs(output_dir, exist_ok=True)
    filenames = load_img_name(input_dir)

    for filename in tqdm(filenames):

        in_name = input_dir +  filename
        out_name = output_dir + filename
        ext = out_name.split('.')[-1]
        save_name = out_name.replace(ext, 'json')


        # Corner detection
        im = cv2.imread(in_name)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp_list = HarrisDetector(gray)

        # Draw key points
        kp_img = np.copy(im)
        for (x, y) in kp_list:
            kp_img = cv2.circle(kp_img, (y,x), 6, (0,0,255), -1)
        cv2.imwrite(out_name, kp_img)

        # Save key points
        with open(save_name, "w") as fp:
            json.dump(kp_list, fp)

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    input_dir = checkSlash(args.input_dir)
    output_dir = checkSlash(args.output_dir)

    detect_all(input_dir, output_dir)

