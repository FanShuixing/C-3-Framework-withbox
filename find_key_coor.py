import math
from itertools import combinations
import cv2
import matplotlib.pyplot as plt
from scipy import io as sio
import numpy as np
import torch


def get_largest_n(arr, N, keypoint_nums):
    '''
    给定arr，在arr中寻找N个最大值当中，最多为keypoint_nums的较大值
    如N=2000,keypoint_nums=200,就是在数组arr中，找出最大的2000个值的索引，
    然后返回这2000个值对应的坐标，根据坐标，可以得到2000个最大值,当然里面有很多重复的，
    所以取set，取了set后的数目是不定的
    '''
    # Convert it into a 1D array
    a_1d = arr.flatten()

    # Find the indices in the 1D array
    idx_1d = a_1d.argsort()[-N:]

    # convert the idx_1d back into indices arrays for each dimension
    x_idx, y_idx = np.unravel_index(idx_1d, arr.shape)
    large_list = []
    # Check that we got the largest values.
    for x, y, in zip(x_idx, y_idx):
        if len(large_list) == keypoint_nums:
            break
        large_list.append(arr[x][y])
        large_list = list(set(large_list))
    return large_list


def find_coor(arr2D, top_value):
    '''
    get_largest_n()已经返回了那些较大的关键点值，
    但是因为一个最大值通常对应非常多的坐标，但是这些坐标其实对应的只是一个关键点，
    所以对应每一个大值，我们取其对应坐标的均值作为其对应关键点的坐标
    '''
    result = np.where(arr2D == top_value)
    listOfCordinates = list(zip(result[0], result[1]))
    listOfCordinates = np.array(listOfCordinates)
    x = np.sum(listOfCordinates[:, 0]) / len(listOfCordinates)
    y = np.sum(listOfCordinates[:, 1]) / len(listOfCordinates)
    return int(x), int(y)


def dist(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_coorindex(nums):
    j_start = 1
    coordinate = []
    for i in range(nums):
        for j in range(j_start, nums):
            coordinate.append([i, j])
        j_start += 1
    coordinate = np.array(coordinate)
    return coordinate


def get_remove_index(arr2D, boxes, coor_dict):
    remove_box_idx = []
    for key in coor_dict.keys():
        max_box_idx = -1
        tmp_max = -1
        for box_idx in coor_dict[key]:
            if arr2D[boxes[box_idx][1], boxes[box_idx][0]] > tmp_max:
                tmp_max = arr2D[boxes[box_idx][1], boxes[box_idx][0]]
                max_box_idx = box_idx
        coor_dict[key].remove(max_box_idx)
        remove_box_idx += coor_dict[key]
    remove_box_idx = np.array(remove_box_idx)
    remove_box_idx = np.unique(remove_box_idx)
    return remove_box_idx


def show_box(final_boxes):
    img = cv2.imread('/output/SHHB_results/flir_20191209T033234_rgb_image_pred_37.0.png')
    img = cv2.resize(img, (768, 576))
    for box in final_boxes:
        x0, y0 = box[0], box[1]
        b_width = 10
        b_height = 10
        color = (255, 0, 0)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + b_width), int(y0 + b_height)), color, 2)
    #     plt.figure(figsize=(12,12))
    #     io.imshow(a)
    cv2.imwrite('tmp.jpg', img)


def get_coor_dict(idx, keypoint_nums):
    coor_dict = {}
    coordinate = get_coorindex(keypoint_nums)
    for each in coordinate[idx]:
        key = each[0]
        if key not in coor_dict:
            coor_dict[key] = [each[1]]
            coor_dict[key].append(key)
        else:
            coor_dict[key].append(each[1])
    return coor_dict


def get_topk(heatmap, width=768):
    '''
    返回heatmap keypoint的关键点，对应的xs,ys坐标。先找到关键点的坐标，然后用x*width+y表示索引
    return:
    xs:[1,box_nums,1]
    ys:[1,box_nums,1]
    inds:[1,box_nums]
    '''
    arr2D = heatmap.cpu().numpy()
    arr2D = np.squeeze(arr2D)
    keypoint_value = get_largest_n(arr2D, 2000, keypoint_nums=200)
    # 得到框，boxes里面有大量的重复的框
    boxes = []
    for top_value in keypoint_value:
        y0, x0 = find_coor(arr2D, top_value)
        boxes.append((x0, y0))
    boxes = np.array(boxes)

    distances = [dist(p1, p2) for p1, p2 in combinations(boxes, 2)]
    distances = np.array(distances)
    idx = np.where(distances < 10)

    coor_dict = get_coor_dict(idx, keypoint_nums=boxes.shape[0])
    remove_box_idx = get_remove_index(arr2D, boxes, coor_dict)
    new_boxes2 = np.delete(boxes, remove_box_idx, axis=0)
    xs = new_boxes2[:, 0]
    ys = new_boxes2[:, 1]
    inds = ys * width + xs
    xs = np.expand_dims(xs, axis=0)
    xs = np.expand_dims(xs, axis=-1)
    ys = np.expand_dims(ys, axis=0)
    ys = np.expand_dims(ys, axis=-1)
    inds = np.expand_dims(inds, axis=0)
    return inds, ys, xs


def get_wh(feat, inds):
    '''
    feat:tensor,[1,2,576,768] ,pred_wh
    inds:tensor,eg:[1,11]
    用numpy的形式根据索引返回pred_wh对应的值
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    dim = feat.size(2)
    inds = torch.from_numpy(inds)
    ind = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
    wh = np.take(feat.cpu().numpy(), ind.numpy())
    return wh


def main():
    den = sio.loadmat('SHHB_results/pred/flir_20191209T033234_rgb_image.mat')
    arr2D = den['data']

    keypoint_value = get_largest_n(arr2D, 2000, keypoint_nums=200)
    # 得到框，boxes里面有大量的重复的框
    boxes = []
    for top_value in keypoint_value:
        y0, x0 = find_coor(arr2D, top_value)
        boxes.append((x0, y0))
    boxes = np.array(boxes)

    distances = [dist(p1, p2) for p1, p2 in combinations(boxes, 2)]
    distances = np.array(distances)
    idx = np.where(distances < 10)

    coor_dict = get_coor_dict(idx, keypoint_nums=boxes.shape[0])
    remove_box_idx = get_remove_index(arr2D, boxes, coor_dict)
    new_boxes2 = np.delete(boxes, remove_box_idx, axis=0)

    show_box(new_boxes2)


if __name__ == '__main__':
    main()