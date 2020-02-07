import os, json
import numpy as np
import scipy.io as sio
import re
import pandas

'''
将目标检测数据制作为 人群计数的mat格式的数据
'''

def main(save_dir):
    label_list = pandas.read_csv('/input0/val_meta.csv').values[:, 0]
    for each in label_list:
        each = each.split('/')[-1]
        name = re.sub('.json', '.mat', each)
        tmp = []
        with open(os.path.join('/input0/mask_labels', each)) as fr:
            info = json.load(fr)
        nums = info['num_box']
        width = info['image_width']
        height = info['image_height']
        for i in range(nums):
            x_center = (info['bboxes'][i]['x_max'] - info['bboxes'][i]['x_min']) / 2 + info['bboxes'][i]['x_min']
            y_center = (info['bboxes'][i]['y_max'] - info['bboxes'][i]['y_min']) / 2 + info['bboxes'][i]['y_min']
            x_center *= width
            y_center *= height
            #             print(x_center,y_center)
            tmp.append([x_center, y_center])
        tmp = np.array(tmp)
        data = {'location': tmp, 'number': float(nums)}
        save = {'image_info': data}
        #         print(save_dir+name)
        sio.savemat(save_dir + name, save)
    print('done!')


if __name__ == '__main__':
    main(save_dir='val_mat/')