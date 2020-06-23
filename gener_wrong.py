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

    diff = abs(info[:, 2] - info[:, 3])
    idx_list = np.where(diff > 0)
    with open('result_report.txt','a+') as fr:
        for idx in idx_list[0]:
            name = info[idx][0].split('/')[-1].replace('.jpg', '')
            ori = os.path.join(args.root_dir, args.meta_file,
                               'images/%s_pred_%s_gt_%s_box_%s.png' % (name, info[idx][1], info[idx][2], info[idx][3]))
            target = os.path.join(args.output_dir, args.meta_file,
                                  '%s_pred%s_gt%s_box_%s.png' % (name, info[idx][1], info[idx][2], info[idx][3]))
            shutil.copy(ori, target)

        w_idx = np.where(diff > 0)[0]
        fr.write('*' * 10+args.meta_file+'*' * 10)
        fr.write('\n\n')
        fr.write('图片数目:错误数/总数=%s/%s' % (len(w_idx), info.shape[0])+ '='+ str(len(w_idx) / info.shape[0]))
        fr.write('\n\n')
#         print('*' * 10, args.meta_file, '*' * 10)
#         print('图片数目:错误数/总数=%s/%s' % (len(w_idx), info.shape[0]), '=', len(w_idx) / info.shape[0])

        w_idx_than_1 = np.where(diff > 1)[0]
        fr.write('图片数目:错误超过1的数目/总数=%s/%s' % (len(w_idx_than_1), info.shape[0])+\
                 '='+str(len(w_idx_than_1) / info.shape[0]))
        fr.write('\n\n\n')
#         print('图片数目:错误超过1的数目/总数=%s/%s' % (len(w_idx_than_1), info.shape[0]), '=', len(w_idx_than_1) / info.shape[0])






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", help='the path to save all box predictions')
    parser.add_argument('--output_dir', help='save the output images of wrong nums large than 1')
    parser.add_argument('--meta_file', help='')

    args = parser.parse_args()
    main(args)