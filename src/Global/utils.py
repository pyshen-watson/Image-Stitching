from os import listdir
from os.path import isdir

def isImage(str):
    exts = ['.jpg','.JPG', '.png', '.PNG', '.bmp', '.BMP', '.jpeg', '.JPEG']
    for ext in exts:
        if ext in str:
            return True
    return False

def load_img_name(img_dir):

    if not isdir(img_dir):
        return []

    return [file for file in listdir(img_dir) if isImage(file)]

def checkSlash(path):
    if path[-1] != '/':
        return path + '/'
    return path
    
import cv2
import numpy as np
from matplotlib import pyplot as plt

def imshow(img):

    channel = img.shape[-1]

    if channel == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")

    plt.show()

def getGaussianKernel(ksize, sigma):
    gaussian1d = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.matmul(gaussian1d, np.transpose(gaussian1d))
    return kernel