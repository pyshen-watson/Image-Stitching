import enum
import cv2
import json
import numpy as np
from cv2 import Mat
from tqdm import tqdm
from os import makedirs
from typing import  List, Tuple
from Global.utils import checkSlash
from Global.envs import OPTIMIZE_REMAIN_RATIO as RATIO
from Global.envs import OPTIMIZE_BLUR_KSIZE as KSIZE
from Global.envs import OPTIMIZE_BLUR_SIGMA as SIGMA

def get_drift_matrix(model_list: List[np.ndarray], sample_img) -> np.ndarray:

    corner = np.zeros((4, 3))

    img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    horizontal_cut = img[img.shape[0]//2, :]
    content = np.nonzero(horizontal_cut)[0]
    left = np.min(content)
    right = np.max(content)

    vertical_cut = img[:, left]
    content = np.nonzero(vertical_cut)[0]
    top = np.min(content)
    bot = np.max(content)


    corner[0] = np.array([top, left, 1])
    corner[1] = np.array([bot, left, 1])
    corner[2] = (model_list[-1] @ np.array([bot, right, 1]))
    corner[3] = (model_list[-1] @ np.array([top, right, 1]))

    new_corner = np.copy(corner)
    new_corner[0] = np.array([0, 0, 1])
    new_corner[1] = np.array([bot-top+1, 0, 1])
    new_corner[2] = np.array([bot-top+1, corner[3,1]-corner[0,1]+1, 1])
    new_corner[3] = np.array([0, corner[3,1]-corner[0,1]+1, 1])

    A = []
    b = []


    for i in range(4):
        x1, y1, = corner[i][0:2]
        x2, y2, = new_corner[i][0:2]

        A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -x2*y1])
        b.append(x2)

        A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y2*y1])
        b.append(y2)


    A = np.array(A)
    b = np.array(b)

    x = np.linalg.solve(A, b)
    x = np.append(x, 1.0)
    x = x.reshape(3,3)

    return x

def crop_black(img: Mat) -> Mat:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y = np.nonzero(gray)

    min_x = np.min(x)
    min_y = np.min(y)
    max_x = np.max(x)
    max_y = np.max(y)

    img = img[min_x:max_x+1, min_y:max_y+1, :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for r in range(img.shape[0]-1, 0, -1):
        cut = gray[r,:]
        content = np.nonzero(cut)[0]
        if content.size / cut.size > RATIO:
            img = img[0:r+1, :, :]
            break

    for c in range(img.shape[1]-1, 0, -1):
        cut = gray[:,c]
        content = np.nonzero(cut)[0]
        if content.size / cut.size > RATIO:
            img = img[:, :c+1, :]
            break

    return img

def optimize(input_dir, output_dir):

    makedirs(output_dir, exist_ok=True)
    sample_img = cv2.imread(f'{input_dir}example.jpg')

    with open(f'{input_dir}model.json', "r") as fp:
        model_list = json.load(fp)
        model_list = [ np.array(model) for model in model_list]



    img = cv2.imread(f'{input_dir}Full.jpg')
    height, width, _ = img.shape

    model = get_drift_matrix(model_list, sample_img)
    img = cv2.transpose(img)
    img = cv2.warpPerspective(img, model, (height, width))
    img = cv2.transpose(img)
    img = crop_black(img)

    img = cv2.GaussianBlur(img, (KSIZE,KSIZE), SIGMA)



    cv2.imwrite(f'{output_dir}Panorama.png', img)
    print(f'Write paranoma at {output_dir}Panorama.png')



if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    input_dir = checkSlash(args.input_dir)
    output_dir = checkSlash(args.output_dir)

    optimize(input_dir, output_dir)