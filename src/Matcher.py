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

    return match_list

def RANSAC(match_list):

    best_model = np.zeros((3,3))
    best_record = 0

    for _ in tqdm(range(N_RANSAC)):


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

    return best_model

def match_all(img_dir, kp_dir, output_dir):

    makedirs(output_dir, exist_ok=True)
    imgNames = load_img_name(img_dir)
    imgNames.sort()
    kpNames = load_json_name(kp_dir)
    kpNames.sort()
    filenames = zip(imgNames, kpNames)


    images = []
    kp_lists = []

    for (imgName, kpName) in filenames:

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

        images.append(im)
        kp_lists.append(kp_list)

    for i in range(7,len(images)-1):

        img1, img2 = images[i], images[i+1]
        kp_list1, kp_list2 = kp_lists[i], kp_lists[i+1]

        matches = match(kp_list1, kp_list2)
        model = RANSAC(matches)

        print(len(matches))

        inlier = []
        for (kp1, kp2) in matches:
            v1 = np.array([kp1['x'], kp1['y'], 1.0])
            v2 = np.array([kp2['x'], kp2['y'], 1.0])
            err =  np.sum((np.matmul(model, v2) - v1) ** 2)
            if err < IT:
                inlier.append((kp1, kp2))

        # for kp in kp_list1:
        #     cv2.circle(img1, (kp['y'], kp['x']), 4, (0,0,255), -1)
        # for kp in kp_list2:
        #     cv2.circle(img2, (kp['y'], kp['x']), 4, (0,0,255), -1)

        # for (kp1, kp2) in matches:
        #     cv2.circle(img1, (kp1['y'], kp1['x']), 6, (0,255,255), -1)
        #     cv2.circle(img2, (kp2['y'], kp2['x']), 6, (0,255,255), -1)

        for (kp1, kp2) in inlier:
            cv2.circle(img1, (kp1['y'], kp1['x']), 6, (0,255,0), -1)
            cv2.circle(img2, (kp2['y'], kp2['x']), 6, (0,255,0), -1)

        cv2.imwrite('img1.jpg', img1)
        cv2.imwrite('img2.jpg', img2)

        break


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

