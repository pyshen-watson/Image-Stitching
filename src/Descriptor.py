import cv2
import json
import numpy as np
from cv2 import Mat
from tqdm import tqdm
from numba import njit
from os import makedirs
from typing import List, Tuple
from Global.utils import load_img_name, load_json_name, getGaussianKernel, checkSlash, print_exit
from Global.envs import DESCRIBE_SLIDE_KSIZE as SKSIZE
from Global.envs import DESCRIBE_GRAD_KSIZE as GKSIZE
from Global.envs import DESCRIBE_BLUR_KSIZE as BKSIZE
from Global.envs import DESCRIBE_BLUR_SIGMA as BSIGMA
from Global.envs import DESCRIBE_PATCH_KSIZE as PKSIZE
from Global.envs import DESCRIBE_PATCH_SIGMA as PSIGMA
from Global.envs import DESCRIBE_PATCH_SMPSIZE as SMPSIZE
from Global.envs import DESCRIBE_BUCKET_NUMBER as N_BUCKET
from Global.envs import DRAW_PEN_SIZE_THIN as PEN_SIZE

class Keypoint:

    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y

    def setGrad(self, mag:float, mj:float) -> None:
        self.mag = mag

        unit = 2 * np.pi / N_BUCKET
        rad = mj * unit
        self.ori = rad * 180 / np.pi
        self.ori_rad = self.ori * np.pi / 180

    def setDesc(self, desc: List[float]):
        self.desc = desc

    def getVector(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        start = (self.y, self.x)

        end_x = int(self.y + self.mag * np.cos(self.ori_rad)/15)
        end_y = int(self.x + self.mag * np.sin(self.ori_rad)/15)
        end = (end_x, end_y)

        return (start, end)

    def dump(self) -> Tuple[int, int, List[float]]:
        return (self.x, self.y, self.desc)


@njit
def vote(window_mag:np.ndarray, window_ori:np.ndarray) -> List[int]:

    unit = np.pi * 2 / N_BUCKET
    buckets = np.zeros(N_BUCKET)

    for row in range(SKSIZE):
        for col in range(SKSIZE):

            mag = window_mag[row, col]
            ori = window_ori[row, col]

            bucket_id = int(((ori + np.pi) // unit).item())
            buckets[bucket_id] += mag

    peak = np.max(buckets)
    ret = list()

    for i in range(N_BUCKET):
        if buckets[i] >= peak * 0.8:
            ret.append(i)

    return ret

def getPatch(img: Mat, x: int, y: int, ori: float) -> Mat:

    Rect = Tuple[Tuple[int, int], Tuple[int, int], float]

    def boundingBox(rect: Rect) -> Rect:

        box = cv2.boxPoints(rect)

        x_max = int(np.max(box[:,0]))
        x_min = int(np.min(box[:,0]))
        y_max = int(np.max(box[:,1]))
        y_min = int(np.min(box[:,1]))


        center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        angle = 0

        return (center, (width, height), angle)

    def cropBox(img: Mat, rect: Rect) -> Mat:
        ((x, y), (w, h), a) = rect
        return img[ y-h//2 : y+h-h//2 , x-w//2 : x+w-w//2]

    def rotateBox(img: Mat, angle: float) -> Mat:

        height, width = img.shape
        center = (width//2, height//2)

        M = cv2.getRotationMatrix2D(center, angle, 1)

        abs_cos = abs(M[0,0])
        abs_sin = abs(M[0,1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        M[0, 2] += bound_w/2 - center[0]
        M[1, 2] += bound_h/2 - center[1]

        img_rot = cv2.warpAffine(img, M, (bound_w, bound_h))

        return img_rot

    rect = ((y,x), (PKSIZE, PKSIZE), ori)

    box = boundingBox(rect)
    box_img = cropBox(img, box)
    box_img = rotateBox(box_img, ori)

    bx, by = box_img.shape[0] // 2, box_img.shape[1] // 2
    H_PKSIZE = PKSIZE // 2

    crop = box_img[bx-H_PKSIZE : bx+(PKSIZE-H_PKSIZE), by-H_PKSIZE : by+(PKSIZE-H_PKSIZE)]

    return crop

def findMajorOrientation(img: Mat, kp_list: List[Keypoint]) -> List[Keypoint]:

    img = cv2.copyMakeBorder(img, SKSIZE, SKSIZE, SKSIZE, SKSIZE, borderType=cv2.BORDER_CONSTANT, value=0)
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=GKSIZE)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=GKSIZE)

    grad_mag = ((Ix**2)+(Iy**2))**0.5
    grad_ori = np.arctan2(Ix, Iy)

    new_kp_list = []

    for kp in kp_list:

        x = kp.x + SKSIZE
        y = kp.y + SKSIZE
        top = x - SKSIZE//2
        left = y - SKSIZE//2

        window_mag = grad_mag[top:top+SKSIZE, left:left+SKSIZE]
        window_ori = grad_ori[top:top+SKSIZE, left:left+SKSIZE]
        window_mag += getGaussianKernel(BKSIZE, BSIGMA)


        major_ori = vote(window_mag, window_ori)

        for mj in major_ori:
            new_kp = Keypoint(kp.x, kp.y)
            new_kp.setGrad(grad_mag[x,y], mj)
            new_kp_list.append(new_kp)

    return new_kp_list

def findDescriptor(img: Mat, kp_list: List[Keypoint]) -> List[Keypoint]:

    new_kp_list = []
    img = cv2.copyMakeBorder(img, PKSIZE, PKSIZE, PKSIZE, PKSIZE, borderType=cv2.BORDER_CONSTANT, value=0)

    for kp in kp_list:

        x, y, ori = kp.x + PKSIZE, kp.y + PKSIZE, kp.ori

        # Crop the neighbor, blur and resize to 8x8
        crop = getPatch(img, x, y, ori)
        crop = crop.astype(np.float64)
        crop *= getGaussianKernel(PKSIZE, PSIGMA)
        crop = cv2.resize(crop, (SMPSIZE, SMPSIZE), interpolation=cv2.INTER_AREA)

        # Normalize the crop
        mu, std = np.mean(crop), np.std(crop)
        crop = (crop - mu) / (std + 1e-8)
        desc = (crop.reshape(SMPSIZE*SMPSIZE)).tolist()

        kp.setDesc(desc)

        new_kp_list.append(kp)

    return new_kp_list

def draw(img: Mat, kp_list :List[Keypoint], out_name:str) -> None:

    for kp in kp_list:

        start, end = kp.getVector()
        box = cv2.boxPoints(((kp.y, kp.x), (PKSIZE, PKSIZE), kp.ori)).astype(np.int0)

        cv2.arrowedLine(img, start, end, (0,0,255), PEN_SIZE)
        cv2.drawContours(img, [box], 0, (0,0,255), PEN_SIZE)

    cv2.imwrite(out_name, img)

def describe_all(img_dir: str, kp_dir:str, output_dir:str) -> None:

    makedirs(output_dir, exist_ok=True)
    imgNames = load_img_name(img_dir)
    imgNames.sort()
    kpNames = load_json_name(kp_dir)
    kpNames.sort()
    filenames = zip(imgNames, kpNames)

    with tqdm(desc='Describing: ', total=len(imgNames)) as pbar:

        for (imgName, kpName) in filenames:

            # Check input
            prefix_img = imgName.split('.')[0]
            prefix_json = kpName.split('.')[0]
            if prefix_img != prefix_json:
                print_exit('Missing image or keypoint data')

            # Setting path
            img_name = img_dir + imgName
            kp_name = kp_dir + kpName
            out_name = output_dir + imgName
            save_name = out_name.replace(out_name.split('.')[-1], 'json')

            # Load resources
            im = cv2.imread(img_name)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            with open(kp_name, "r") as fp:
                kp_list = json.load(fp)
                kp_list = [ Keypoint(kp[0], kp[1]) for kp in kp_list]


            # Describe keypoints
            kp_list = findMajorOrientation(gray, kp_list)
            kp_list = findDescriptor(gray, kp_list)
            draw(im, kp_list, out_name)

            # Save description
            with open(save_name, "w") as fp:
                kp_list = [ kp.dump() for kp in kp_list]
                json.dump(kp_list, fp)

            pbar.update(1)

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



