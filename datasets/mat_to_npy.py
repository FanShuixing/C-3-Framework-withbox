import os, cv2
import scipy.io as scio
import numpy as np
from tqdm import tqdm
import numpy as np
import math
import json

standard_size=[576,768];

def get_density(img_dir,det_gt_dir,w=768,h=576):

    mat=get_mat(det_gt_dir)
    im = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    annPoints = mat['image_info']['location']

    rate = standard_size[0] / h;
    rate_w = w * rate;
    if rate_w > standard_size[1]:
        rate = standard_size[1] / w;
    rate_h = float(int(h * rate)) / h;
    rate_w = float(int(w * rate)) / w;
    im = cv2.resize(im, (int(w * rate), int(h * rate)));
    annPoints[:, 0] = annPoints[:, 0] * float(rate_w)
    annPoints[:, 1] = annPoints[:, 1] * float(rate_h)

    im_density = get_density_map_gaussian(im, annPoints)
    return im_density


def get_mat(det_gt_dir,width=768,height=576):
    tmp=[]
    with open(det_gt_dir) as fr:
        info = json.load(fr)
    nums = info['num_box']
    for i in range(nums):
        x_center = (info['bboxes'][i]['x_max'] - info['bboxes'][i]['x_min']) / 2 + info['bboxes'][i]['x_min']
        y_center = (info['bboxes'][i]['y_max'] - info['bboxes'][i]['y_min']) / 2 + info['bboxes'][i]['y_min']
        x_center *= width
        y_center *= height
        tmp.append([x_center, y_center])
    tmp = np.array(tmp)
    data = {'location': tmp, 'number': float(nums)}
    save = {'image_info': data}
    return save






def matlab_style_gauss2D(shape=(300, 300), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_density_map_gaussian(im, points):
    im_density = np.zeros(im.shape)
    [h, w] = im_density.shape

    for j in range(0, len(points)):
        f_sz = 15
        sigma = 4.0
        # H = matlab.fspecial('Gaussian', [f_sz, f_sz], sigma)
        H = matlab_style_gauss2D([f_sz, f_sz], sigma)
        x = min(w, max(1, abs(int(math.floor(points[j, 0])))))
        y = min(h, max(1, abs(int(math.floor(points[j, 1])))))

        if x > w or y > h:
            continue
        x1 = x - int(np.floor(f_sz / 2))
        y1 = y - int(np.floor(f_sz / 2))
        x2 = x + int(np.floor(f_sz / 2))
        y2 = y + int(np.floor(f_sz / 2))
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
        if x1 < 1:
            dfx1 = abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dfy1 = abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1 + dfx1
        y1h = 1 + dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2
        if change_H:
            # H = matlab.fspecial('Gaussian', [double(y2h - y1h + 1), double(x2h - x1h + 1)], sigma)
            H = matlab_style_gauss2D([float(y2h - y1h + 1), float(x2h - x1h + 1)], sigma)
        im_density[y1 - 1: y2, x1 - 1: x2] = im_density[y1 - 1: y2, x1 - 1: x2] + H
    return im_density


if __name__ == '__main__':
    get_density('/input0/normal/images/flir_20191209T221444_rgb_image.jpg','/input0/normal/mask_labels/flir_20191209T221444_rgb_image.json')
