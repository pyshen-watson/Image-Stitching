import numpy as np
import random
# second closest neighbor

def SSD(img1, x1, y1, img2, x2, y2):
    window1 = img1[x1-2:x1+3, y1-2:y1+3]
    window2 = img2[x2-2:x2+3, y2-2:y2+3]
    return np.sum((window1 - window2)**2)

def feature_matching(img1, img2, kp1, kp2):
    match = []
    for (x1, y1, _) in kp1:
        ssd_list = []
        for (x2, y2, _) in kp2:
            ssd_list.append((x2, y2, SSD(img1, x1, y1, img2, x2, y2)))
        ssd_list.sort(key=lambda s:s[3], reverse=True)
        first = ssd_list[0]
        second = ssd_list[1]
        if first[2]/second[2] < 0.8:
            match.append((x1, y1, first[0], first[1]))
    return match

# RANSAC
def RANSAC(match):
    new_match = []
    Greatest_H = np.array((3, 3))
    largestInliersN = 0
    for _ in range(500):
        rand_select = random.sample(match, 4)
        A = []
        b = np.mat('0, 0, 0, 0, 0, 0, 0, 0, 0').T
        for (x1, y1, x2, y2) in rand_select:
            A.append((x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2))
            A.append((0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2))
        A = np.mat(A)
        X = np.linalg.solve(A, b).tolist()
        H = np.array([X[0, 0], X[0, 1], X[0, 2]], [X[0, 3], X[0, 4], X[0, 5]], [X[0, 6], X[0, 7], X[0, 8]])
        inliersN = 0
        for (x1, y1, x2, y2) in match:
            flag = 0
            for i in range(4):
                if rand_select[i, 0] == x1 and rand_select[i, 1] == y1 and rand_select[i, 2] == x2 and rand_select[i, 3] == y2:
                    flag = 1
            if flag == 0:
                mXY = np.array([x1] , [y1], [1])
                generateXY = H.dot(mXY)
                generateXY = generateXY / generateXY[0, 2]
                if ((generateXY[0, 0] - x2)**2 + (generateXY[0, 1] - y2)**2) < 10:
                    inliersN += 1
        if inliersN > largestInliersN:
            largestInliersN = inliersN
            Greatest_H = H
    for (x1, y1, x2, y2) in match:
        mXY = np.array([x1], [y1], [1])
        generateXY = Greatest_H.dot(mXY)
        generateXY = generateXY / generateXY[0, 2]
        if ((generateXY[0, 0] - x2)**2 + (generateXY[0, 1] - y2)**2) < 10:
            new_match.append(x1, y1, x2, y2)
    return Greatest_H, new_match