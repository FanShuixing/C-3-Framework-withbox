import pandas as pd
import shutil
import os
import numpy as np
import argparse

'''
将数据集按照正常，倾斜，遮挡划分来分布进行预测
'''


def main(args):
    type_name = args.meta_file.split('_')[-1]
    with open(os.path.join(args.root_dir, args.meta_file, type_name + '.csv')) as fr:
        info = pd.read_csv(fr).values
    os.makedirs(os.path.join(args.output_dir, args.meta_file), exist_ok=True)

    diff = abs(info[:, 1] - info[:, 2])
    idx_list = np.where(diff > 1)
    for idx in idx_list[0]:
        name = info[idx][0].split('/')[-1].replace('.jpg', '')
        ori = os.path.join(args.root_dir, args.meta_file,
                           'images/%s_pred_%s_gt_%s.png' % (name, info[idx][1], info[idx][2]))
        target = os.path.join(args.output_dir, args.meta_file,
                              '%s_pred%s_gt%s.png' % (name, info[idx][1], info[idx][2]))
        shutil.copy(ori, target)

    w_idx = np.where(abs(info[:, 1] - info[:, 2]) > 0)[0]
    print('*' * 10, args.meta_file, '*' * 10)
    print('图片数目:错误数/总数=%s/%s' % (len(w_idx), info.shape[0]), '=', len(w_idx) / info.shape[0])
    w_idx_than_1 = np.where(abs(info[:, 1] - info[:, 2]) > 1)[0]
    print('图片数目:错误超过1的数目/总数=%s/%s' % (len(w_idx_than_1), info.shape[0]), '=', len(w_idx_than_1) / info.shape[0])

    idx = info[:, 1] - info[:, 2]
    err_more_idx = np.where(idx > 0)[0]
    a = (info[err_more_idx][:, 1] - info[err_more_idx][:, 2]) / (info[err_more_idx][:, 2])
    if len(a) == 0:
        print("木材根数:根数差值大于0的错误率统计:", 0)
    else:
        print("木材根数:根数差值大于0的错误率统计:", np.sum(a) / len(a))

    err_less_idx = np.where(idx < 0)[0]
    a = abs(info[err_less_idx][:, 1] - info[err_less_idx][:, 2]) / (info[err_less_idx][:, 2])
    print("木材根数:根数差值小于0的错误率统计:", np.sum(a) / len(a))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", help='the path to save all box predictions')
    parser.add_argument('--output_dir', help='save the output images of wrong nums large than 1')
    parser.add_argument('--meta_file', help='')

    args = parser.parse_args()
    main(args)