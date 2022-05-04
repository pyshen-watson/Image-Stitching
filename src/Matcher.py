from __future__ import annotations
import cv2
import json
import numpy as np
from os import makedirs
from numpy.random import choice
from tqdm import tqdm
from cv2 import Mat
from typing import List, Optional, Tuple
from numba import njit
from numba.typed import List as nblist
from Global.utils import load_img_name, load_json_name, checkSlash, print_exit
from Global.envs import MATCH_ERROR_THRESHOLD as ET
from Global.envs import MATCH_INLIER_THRESHOLD as IT
from Global.envs import MATCH_RANSAC_NUMBER as N_RANSAC
from Global.envs import DRAW_PEN_SIZE as PEN_SIZE
from Global.envs import DRAW_PEN_SIZE_THIN as PEN_THIN


class Keypoint:

    def __init__(self, x:int, y:int,  dv:Optional[List[float]] = None):
        self.x = x
        self.y = y

        if dv:
            self.dv = np.array(dv)

    def tolist(self) -> List:
        return [self.x, self.y]

KPpair = Tuple[Keypoint, Keypoint]

class Image:

    def __init__(self, name:str, img:Mat, kp_list:List[Keypoint]):
        self.name = name
        self.img = img
        self.kp_list = kp_list

        self.model = np.identity(3)
        self.inlier_list = []

def RANSAC(match_list:List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray,List[KPpair]]:


    @njit
    def fit(match_list: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:

        best_model = np.zeros((3,3))
        best_record = [(0,0,0,0) for i in range(0)]

        for _ in range(N_RANSAC):

            # random match a model

            A = [] # 6x6 matrix
            b = [] # 6x1 vector

            index = choice(len(match_list), 3)

            for i in index:
                (x1, y1, x2, y2) = match_list[i]
                A.append([x2, y2, 1, 0, 0, 0])
                A.append([0, 0, 0, x2, y2, 1])
                b.append(x1)
                b.append(y1)

            A = np.array(A, dtype=np.float64)
            b = np.array(b, dtype=np.float64)

            if np.linalg.det(A) == 0:
                continue

            x = np.linalg.inv(A) @ b

            # evaluate this random model
            model = np.concatenate((x, np.array([0,0,1]))).reshape((3,3))

            inliner_list = []
            for (x1, y1, x2, y2) in match_list:

                v1 = np.array([x1, y1, 1], dtype=np.float64)
                v2 = np.array([x2, y2, 1], dtype=np.float64)

                err =  np.sum(( (model @ v2) - v1) ** 2)

                if err < IT:
                    inliner_list.append((x1, y1, x2, y2))

            if len(inliner_list) > len(best_record):
                best_model = model
                best_record = inliner_list

        return best_model, best_record

    best_model, best_record = fit(nblist(match_list))
    best_record = [(Keypoint(x1, y1), Keypoint(x2, y2)) for (x1, y1, x2, y2) in best_record]

    return best_model, best_record

def img_match(img1: Image, img2:Image) -> List[Tuple[int, int, int, int]]:

    kp_list1 = [(kp.x, kp.y, kp.dv) for kp in img1.kp_list]
    kp_list2 = [(kp.x, kp.y, kp.dv) for kp in img2.kp_list]

    KPList = List[Tuple[int, int, np.ndarray]]

    @njit
    def match(kp_list1: KPList, kp_list2: KPList) -> List[Tuple[int, int, int, int]]:

        match_list = list()

        for (x1, y1, dv1) in kp_list1:

            score_list = [ (x2, y2, np.sum((np.subtract(dv1, dv2))**2)) for (x2, y2, dv2) in kp_list2]
            score_list.sort(key=lambda s:s[-1])

            first = score_list[0]
            second = score_list[1]

            if first[-1] / second[-1] < ET:
                match_list.append((x1, y1, first[0], first[1]))

        return match_list

    match_list = match(nblist(kp_list1), nblist(kp_list2))

    return match_list

def load_resource(img_dir, kp_dir) -> List[Image]:

    imgNames = load_img_name(img_dir)
    kpNames = load_json_name(kp_dir)

    imgNames.sort()
    kpNames.sort()

    img_pool = []

    for (imgName, kpName) in zip(imgNames, kpNames):

        # Check input
        prefix_img = imgName.split('.')[0]
        prefix_json = kpName.split('.')[0]
        if prefix_img != prefix_json:
            print_exit('Missing image or keypoint data')

        # Setting path
        img_name = img_dir + imgName
        kp_name = kp_dir + kpName

        # Loading resource
        im = cv2.imread(img_name)
        with open(kp_name, "r") as fp:
            kp_list = json.load(fp)

        kp_list = [Keypoint(kp[0], kp[1], kp[2]) for kp in kp_list]
        img_pool.append(Image(imgName, im, kp_list))

    return img_pool

def draw(img_seq: List[Image], out_dir: str) -> None:

    for img in img_seq:

        for kp in img.kp_list:
            cv2.circle(img.img, (kp.y, kp.x), PEN_THIN, (0,255,255), -1)

        for kp in img.inlier_list:
            cv2.circle(img.img, (kp.y, kp.x), PEN_SIZE, (0,0,255), -1)

        cv2.imwrite(out_dir+img.name, img.img)

def match_all(img_dir: str, kp_dir: str, output_dir: str) -> None:

    makedirs(output_dir, exist_ok=True)
    img_seq = load_resource(img_dir, kp_dir)
    img_seq.append(img_seq[0])

    with tqdm(total=(len(img_seq)-1), desc='Matching: ') as pbar:
        for i in range(len(img_seq)-1):
            img1 = img_seq[i]
            img2 = img_seq[i+1]

            model, inlier_list = RANSAC(img_match(img1, img2))

            img1.inlier_list += [kp for (kp, _) in inlier_list]
            img2.inlier_list += [kp for (_, kp) in inlier_list]

            img2.model = model
            pbar.update(1)

    draw(img_seq, output_dir)

    # save (name, model) pair
    with open(f'{output_dir}model.json', "w") as fp:
        model_data = [ (img.name, img.model.tolist()) for img in img_seq]
        json.dump(model_data, fp)

    with open(f'{output_dir}calibration.json', "w") as fp:
        calibration = [kp.tolist() for kp in img_seq[0].kp_list]
        json.dump(calibration, fp)

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

    match_all(img_dir, kp_dir, output_dir)

