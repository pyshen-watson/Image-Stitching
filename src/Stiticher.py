import cv2
import json
import numpy as np
from tqdm import tqdm
from cv2 import Mat
from numba import njit
from os import makedirs
from Global.utils import imshow
from typing import Dict, List, Tuple
from Global.utils import checkSlash

class Image:

    def __init__(self, path, model):
        self.img = cv2.imread(path)
        self.model = np.array(model)
        self.vertices = self.getVertice()

    def getVertice(self) -> List[Tuple[int, int]]:

        vertice = []

        height, width, _ = self.img.shape
        vertice.append((0, 0))
        vertice.append((height, 0))
        vertice.append((0, width))
        vertice.append((height, width))

        return vertice

    def getTransVertice(self) -> List[Tuple[int, int]]:

        vt = []
        for v in self.vertices:

            vec = np.array([v[0], v[1], 1])
            vec_t = (self.model @ vec).astype(np.int0)
            v_t = vec_t.tolist()[:-1]
            vt.append(v_t)

        # print(vt.__str__().replace('[','(').replace(']', ')')[1:-1])
        return vt

    def getTransImage(self, canvas_height, canvas_width):

        im = cv2.transpose(self.img)
        new_img = cv2.warpPerspective(im, self.model, (canvas_height, canvas_width))
        return cv2.transpose(new_img)

def color_map(img: Mat):

    color_overlap = img[:,:,0]+ img[:,:,1] + img[:,:,2]
    color_overlap = np.bool_(color_overlap)
    return color_overlap

def linear_mask2d(overlap, left:int, right:int, top:int, bot:int) -> Tuple[np.ndarray, np.ndarray]:
    mask_1d = np.arange(right-left+1)
    mask_2d = np.tile(mask_1d, (bot-top+1, 1))

    kernel_left = mask_2d / (left-right) + 1
    kernel_right = mask_2d / (right-left)

    mask_left = np.copy(overlap).astype(np.float64)
    mask_left[top:bot+1, left:right+1] *= kernel_left
    mask_left += np.logical_not(overlap).astype(np.float64)
    mask_left = np.expand_dims(mask_left, 2)
    mask_left = np.repeat(mask_left, 3, axis=2)

    mask_right = np.copy(overlap).astype(np.float64)
    mask_right[top:bot+1, left:right+1] *= kernel_right
    mask_right += np.logical_not(overlap).astype(np.float64)
    mask_right = np.expand_dims(mask_right, 2)
    mask_right = np.repeat(mask_right, 3, axis=2)

    return mask_left, mask_right

def draw_image(canvas_height: int, canvas_width: int, img_seq: List[Image]):

    canvas = np.zeros((canvas_height, canvas_width, 3))

    with tqdm(total=len(img_seq), desc='Stitching: ') as pbar:

        for img in img_seq:

            post = img.getTransImage(canvas_height, canvas_width)

            canvas_colored = color_map(canvas)
            post_colored = color_map(post)

            overlap = np.logical_and(canvas_colored, post_colored)

            x, y = np.nonzero(overlap)

            if x.size > 0:

                left = np.min(y)
                right = np.max(y)
                top = np.min(x)
                bot = np.max(x)

                mask_canvas, mask_post = linear_mask2d(overlap, left, right, top, bot)
                canvas = (canvas.astype(np.float64) * mask_canvas).astype(np.uint8)
                post = (post.astype(np.float64) * mask_post).astype(np.uint8)


            canvas += post
            pbar.update(1)

    return canvas

def stitch_all(img_dir, data_dir, output_dir):

    makedirs(output_dir, exist_ok=True)
    with open(f'{data_dir}model.json', "r") as fp:
        model_list = json.load(fp)
        img_seq = [Image(img_dir+name, model) for (name, model) in model_list]
        img_seq[0].model = np.identity(3)

    for i in range(1,len(img_seq)):
        last = img_seq[i-1]
        cur = img_seq[i]
        cur.model = last.model @ cur.model

    all_vertices = []
    for img in img_seq:
        vertices = img.getTransVertice()
        for v in vertices:
            all_vertices.append(v)

    all_vertices = np.array(all_vertices)
    max_x = np.max(all_vertices[:,0])
    min_x = np.min(all_vertices[:,0])
    max_y = np.max(all_vertices[:,1])
    min_y = np.min(all_vertices[:,1])

    canvas_h = max_x
    canvas_w = max_y
    # print(f'x: {min_x}~{max_x}  y: {min_y}~{max_y}')
    # return

    canvas = draw_image(canvas_h, canvas_w, img_seq[:-1])
    cv2.imwrite(f'{output_dir}Full.jpg', canvas)



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

    stitch_all(img_dir, kp_dir, output_dir)