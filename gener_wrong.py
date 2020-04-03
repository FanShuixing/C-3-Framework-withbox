import pandas as pd
import shutil
import os
import numpy as np


def main():
    with open('/output/tf_dir/final.csv') as fr:
        info = pd.read_csv(fr)
    os.makedirs('../wrong', exist_ok=True)
    diff = abs(info.values[:, 1] - info.values[:, 2])
    idx_list = np.where(diff > 1)
    for idx in idx_list[0]:
        name = info.values[idx][0].split('/')[-1].replace('.jpg', '')
        shutil.copy('../SHHB_results/%s_pred_%s_gt_%s.png' % (name, info.values[idx][1],info.values[idx][2]),
                    '../wrong/%s_pre%s_gt%s.png' % (name, info.values[idx][1], info.values[idx][2]))

    with open('/output/tf_dir/final.csv') as fr:
        info = pd.read_csv(fr).values
    w_idx = np.where(abs(info[:, 1] - info[:, 2]) > 0)[0]
    print('错误数/总数=%s/%s' % (len(w_idx), info.shape[0]), '=', len(w_idx) / info.shape[0])
    w_idx_than_1 = np.where(abs(info[:, 1] - info[:, 2]) > 1)[0]
    print('错误超过1的数目/总数=%s/%s' % (len(w_idx_than_1), info.shape[0]), '=', len(w_idx_than_1) / info.shape[0])

    idx = info[:, 1] - info[:, 2]
    err_more_idx = np.where(idx > 0)[0]
    a = (info[err_more_idx][:, 1] - info[err_more_idx][:, 2]) / (info[err_more_idx][:, 2])
    print("根数差值大于0的错误率统计:", np.sum(a) / len(a))

    err_less_idx = np.where(idx < 0)[0]
    a = abs(info[err_less_idx][:, 1] - info[err_less_idx][:, 2]) / (info[err_less_idx][:, 2])
    print("根数差值小于0的错误率统计:", np.sum(a) / len(a))


if __name__ == '__main__':
    main()