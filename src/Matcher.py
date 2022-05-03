import cv2
import json
import numpy as np
from tqdm import tqdm
from random import sample
from os import makedirs
from Global.utils import load_img_name, load_json_name, checkSlash
from Global.envs import MATCH_ERROR_THRESHOLD as ET
from Global.envs import MATCH_INLIER_THRESHOLD as IT
from Global.envs import MATCH_RANSAC_NUMBER as N_RANSAC

def match(kp_list1, kp_list2):

    match_list = []

    for kp1 in kp_list1:

        score_list = [(kp2, np.sum((kp1['desc'] - kp2['desc'])**2)) for kp2 in kp_list2]
        score_list.sort(key=lambda s:s[1])
        first = score_list[0]
        second = score_list[1]
        if first[1] / second[1] < ET:
            match_list.append((kp1, first[0]))

    return match_list, len(match_list)

def RANSAC(match_list):

    best_model = np.zeros((3,3))
    best_record = 0

    for _ in range(N_RANSAC):

        # random match a model

        A = [] # 6x6 matrix
        b = [] # 6x1 vector
        samples = sample(match_list, 3)
        for (kp1, kp2) in samples:
            x1, y1 = kp1['x'], kp1['y']
            x2, y2 = kp2['x'], kp2['y']
            A.append([x2, y2, 1, 0, 0, 0])
            A.append([0, 0, 0, x2, y2, 1])
            b.append(x1)
            b.append(y1)

        A = np.array(A)
        b = np.array(b)
        if np.linalg.det(A) == 0:
            continue

        x = np.linalg.solve(A, b)

        # evaluate this random model

        model = np.concatenate((x, np.array([0,0,1]))).reshape((3,3))

        N_inlier = 0
        for (kp1, kp2) in match_list:
            v1 = np.array([kp1['x'], kp1['y'], 1.0])
            v2 = np.array([kp2['x'], kp2['y'], 1.0])
            err =  np.sum((np.matmul(model, v2) - v1) ** 2)
            if err < IT:
                N_inlier += 1

        if N_inlier > best_record:
            best_record = N_inlier
            best_model = model

    return best_model, best_record

def load_resource(imgNames, kpNames):

    img_pool = []

    for (imgName, kpName) in zip(imgNames, kpNames):

        # Check input
        prefix_img = imgName.split('.')[0]
        prefix_json = kpName.split('.')[0]
        if prefix_img != prefix_json:
            print('Missing image or keypoint data')
            exit()

        # Setting path
        img_name = img_dir + imgName
        kp_name = kp_dir + kpName

        # Loading resource
        im = cv2.imread(img_name)

        with open(kp_name, "r") as fp:
            kp_list = json.load(fp)
        for kp in kp_list:
            kp['desc'] = np.array(kp['desc'])

        img_pool.append((imgName, im, kp_list))

    return img_pool

def sort_pool(img_pool):

    cur = img_pool.pop(0)
    img_sequence = []

    while img_pool:

        (name1, im1, kp_list1) = cur

        best_inline_n = 0
        best_index = 0

        for i, (name2, im2, kp_list2) in tqdm(enumerate(img_pool), desc=f'Last {len(img_pool)}: '):

            match_list, _ = match(kp_list1, kp_list2)
            _, inlier_n = RANSAC(match_list)

            if inlier_n > best_inline_n:
                best_inline_n = inlier_n
                best_index = i

        img_sequence.append(cur)
        cur = img_pool.pop(best_index)

    img_sequence.append(cur)
    return img_sequence

def match_all(img_dir, kp_dir, output_dir, sort=False, draw=False):

    makedirs(output_dir, exist_ok=True)

    imgNames = load_img_name(img_dir)
    imgNames.sort()

    kpNames = load_json_name(kp_dir)
    kpNames.sort()

    img_pool = load_resource(imgNames, kpNames)
    img_sequence = sort_pool(img_pool) if sort else img_pool


    last = img_sequence.pop(0)
    img_sequence.append(last)
    model_list = [np.identity(3)]

    for (name2, im2, kp_list2) in tqdm(img_sequence, total=len(img_sequence), desc='Matching: '):

        (name1, im1, kp_list1) = last
        match_list, _ = match(kp_list1, kp_list2)
        model, _ = RANSAC(match_list)
        model_list.append(model)

        if draw:

            inlier = []
            for (kp1, kp2) in match_list:
                v1 = np.array([kp1['x'], kp1['y'], 1.0])
                v2 = np.array([kp2['x'], kp2['y'], 1.0])
                err =  np.sum((np.matmul(model, v2) - v1) ** 2)
                if err < IT:
                    inlier.append((kp1, kp2))

            for (kp1, kp2) in match_list:
                cv2.circle(im1, (kp1['y'], kp1['x']), 6, (0,255,255), -1)
                cv2.circle(im2, (kp2['y'], kp2['x']), 6, (0,255,255), -1)

            for (kp1, kp2) in inlier:
                cv2.circle(im1, (kp1['y'], kp1['x']), 6, (0,0,255), -1)
                cv2.circle(im2, (kp2['y'], kp2['x']), 6, (0,0,255), -1)

        last = (name2, im2, kp_list2)

    if draw:
        for (name, im, kp_list) in img_sequence:
            out_name = output_dir + name
            cv2.imwrite(out_name, im)


    imgNames.append(imgNames[0])
    first_img_kp_list = [(kp['x'], kp['y']) for kp in last[2]]
    model_data = [ (img_name, model.tolist()) for img_name, model in zip(imgNames, model_list)]

    with open(f'{output_dir}model.json', "w") as fp:
        json.dump(model_data, fp)
    with open(f'{output_dir}first_img_kp_list.json', "w") as fp:
        json.dump(first_img_kp_list, fp)

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

