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
        print(info.values[idx])
        name = info.values[idx][0].split('/')[-1].replace('.jpg', '')
        shutil.copy('../SHHB_results/%s_pred_%s.png' % (name, info.values[idx][1]),
                    '../wrong/%s_pre%s_gt%s.png' % (name, info.values[idx][1], info.values[idx][2]))


if __name__ == '__main__':
    main()