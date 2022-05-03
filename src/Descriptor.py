import cv2
import json
import numpy as np
from tqdm import tqdm
from math import floor
from numba import njit
from os import makedirs
from Global.utils import load_img_name, load_json_name, getGaussianKernel, checkSlash
from Global.box import boundingBox, cropBox, rotateBox
from Global.envs import DESCRIBE_SLIDE_KSIZE as SKSIZE
from Global.envs import DESCRIBE_GRAD_KSIZE as GKSIZE
from Global.envs import DESCRIBE_BLUR_KSIZE as BKSIZE
from Global.envs import DESCRIBE_BLUR_SIGMA as BSIGMA
from Global.envs import DESCRIBE_PATCH_KSIZE as PKSIZE
from Global.envs import DESCRIBE_PATCH_SIGMA as PSIGMA
from Global.envs import DESCRIBE_PATCH_SMPSIZE as SMPSIZE
from Global.envs import DESCRIBE_BUCKET_NUMBER as N_BUCKET

@njit
def vote(window_mag, window_ori):

    unit = np.pi * 2 / N_BUCKET
    buckets = np.zeros(N_BUCKET)

    for row in range(SKSIZE):
        for col in range(SKSIZE):

            mag = window_mag[row, col]
            ori = window_ori[row, col]

            bucket_id = int(((ori + np.pi) // unit).item())
            buckets[bucket_id] += mag

    peak = np.max(buckets)
    ret = []

    for i in range(N_BUCKET):
        if buckets[i] >= peak * 0.8:
            ret.append(i)

    return ret

def mj2ori(mj):
    unit = 2 * np.pi / N_BUCKET
    rad = mj * unit
    deg = rad * 180 / np.pi
    return deg

def getPatch(img, x, y, ori):

    rect = ((y,x), (PKSIZE, PKSIZE), ori)

    box = boundingBox(rect)
    box_img = cropBox(img, box)
    box_img = rotateBox(box_img, ori)
    bx, by = box_img.shape[0] // 2, box_img.shape[1] // 2
    H_PKSIZE = PKSIZE // 2
    crop = box_img[bx-H_PKSIZE : bx+(PKSIZE-H_PKSIZE), by-H_PKSIZE : by+(PKSIZE-H_PKSIZE)]

    return crop

def findMajorOrientation(img, kp_list):

    img = cv2.copyMakeBorder(img, SKSIZE, SKSIZE, SKSIZE, SKSIZE, borderType=cv2.BORDER_CONSTANT, value=0)
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=GKSIZE)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=GKSIZE)

    grad_mag = ((Ix**2)+(Iy**2))**0.5
    grad_ori = np.arctan2(Ix, Iy)

    new_kp_list = []

    for (kp_x,kp_y) in kp_list:

        x = kp_x+SKSIZE
        y = kp_y+SKSIZE
        top = x - SKSIZE//2
        left = y - SKSIZE//2

        window_mag = grad_mag[top:top+SKSIZE, left:left+SKSIZE]
        window_ori = grad_ori[top:top+SKSIZE, left:left+SKSIZE]
        window_mag += getGaussianKernel(BKSIZE, BSIGMA)


        major_ori = vote(window_mag, window_ori)
        for mj in major_ori:

            kp = {
                'x': x,
                'y': y,
                'mag': grad_mag[x,y],
                'ori': mj2ori(mj),
            }

            new_kp_list.append(kp)

    return new_kp_list

def findDescriptor(img, kp_list):

    new_kp_list = []
    img = cv2.copyMakeBorder(img, PKSIZE, PKSIZE, PKSIZE, PKSIZE, borderType=cv2.BORDER_CONSTANT, value=0)

    for kp in kp_list:

        x, y, ori = kp['x']+PKSIZE, kp['y']+PKSIZE, kp['ori']

        # Crop the neighbor, blur and resize to 8x8
        crop = getPatch(img, x, y, ori)
        crop = crop.astype(np.float64)
        crop *= getGaussianKernel(PKSIZE, PSIGMA)
        crop = cv2.resize(crop, (SMPSIZE, SMPSIZE), interpolation=cv2.INTER_AREA)

        # Normalize the crop
        mu, std = np.mean(crop), np.std(crop)
        crop = (crop - mu) / (std + 1e-8)
        desc = (crop.reshape(SMPSIZE*SMPSIZE)).tolist()




        kp.setdefault('desc', desc)

        new_kp_list.append(kp)

    return new_kp_list

def describe_all(img_dir, kp_dir, output_dir):

    makedirs(output_dir, exist_ok=True)
    imgNames = load_img_name(img_dir)
    imgNames.sort()
    kpNames = load_json_name(kp_dir)
    kpNames.sort()
    filenames = zip(imgNames, kpNames)

    for (imgName, kpName) in tqdm(filenames, desc='Describing: ', total=len(imgNames)):

        # Check input
        prefix_img = imgName.split('.')[0]
        prefix_json = kpName.split('.')[0]
        if prefix_img != prefix_json:
            print('Missing image or keypoint data')
            exit()

        # Setting path
        img_name = img_dir + imgName
        kp_name = kp_dir + kpName
        out_name = output_dir + imgName
        ext = out_name.split('.')[-1]
        save_name = out_name.replace(ext, 'json')


        # Load resources
        im = cv2.imread(img_name)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        with open(kp_name, "r") as fp:
            kp_list = json.load(fp)


        # Describe keypoints
        kp_list = findMajorOrientation(gray, kp_list)
        kp_list = findDescriptor(gray, kp_list)

        # Draw gradient and crop box
        kp_img = np.copy(im)
        for kp in kp_list:
            x, y, mag, ori = kp['x'], kp['y'], kp['mag'], kp['ori']

            start = (y, x)
            end = (floor(y + mag*np.cos(ori*np.pi/180)/15), floor(x + mag*np.sin(ori*np.pi/180)/15))
            box = cv2.boxPoints(((y, x), (PKSIZE, PKSIZE), ori)).astype(np.int0)

            cv2.arrowedLine(kp_img, start, end, (0,0,255), 2)
            cv2.drawContours(kp_img, [box], 0, (0,0,255), 2)
        cv2.imwrite(out_name, kp_img)


        with open(save_name, "w") as fp:
            json.dump(kp_list, fp)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("img_dir", type=str)
    parser.add_argument("kp_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    img_dir = checkSlash(args.img_dir)
    kp_dir = checkSlash(args.kp_dir)
    output_dir = checkSlash(args.output_dir)

    describe_all(img_dir, kp_dir, output_dir)



