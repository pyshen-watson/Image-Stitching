import cv2
import numpy as np

def boundingBox(rect):

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

def cropBox(img, rect):
    ((x, y), (w, h), a) = rect
    return img[ y-h//2 : y+h-h//2 , x-w//2 : x+w-w//2]

def rotateBox(img, angle):

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