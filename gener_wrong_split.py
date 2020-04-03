import pandas as pd
import shutil
import os
import numpy as np

'''
将数据集按照正常，倾斜，遮挡划分来分布进行预测

'''

def main(file_dir,status,pred_dir='../result',save_dir='../wrong'):
    with open(file_dir) as fr:
        info = pd.read_csv(fr)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,status),exist_ok=True)

    diff = abs(info.values[:, 1] - info.values[:, 2])
    idx_list = np.where(diff > 1)
    for idx in idx_list[0]:
        name = info.values[idx][0].split('/')[-1].replace('.jpg', '')
        ori=os.path.join(pred_dir,'%s_pred_%s_gt_%s.png'%(name, info.values[idx][1],info.values[idx][2]))
        target=os.path.join(save_dir,status,'%s_pred%s_gt%s.png'%(name,info.values[idx][1], info.values[idx][2]))
        shutil.copy(ori,target)

    with open(file_dir) as fr:
        info = pd.read_csv(fr).values
    w_idx = np.where(abs(info[:, 1] - info[:, 2]) > 0)[0]
    print('*'*10,status,'*'*10)
    print('图片数目:错误数/总数=%s/%s' % (len(w_idx), info.shape[0]), '=', len(w_idx) / info.shape[0])
    w_idx_than_1 = np.where(abs(info[:, 1] - info[:, 2]) > 1)[0]
    print('图片数目:错误超过1的数目/总数=%s/%s' % (len(w_idx_than_1), info.shape[0]), '=', len(w_idx_than_1) / info.shape[0])

    idx = info[:, 1] - info[:, 2]
    err_more_idx = np.where(idx > 0)[0]
    a = (info[err_more_idx][:, 1] - info[err_more_idx][:, 2]) / (info[err_more_idx][:, 2])
#     print(a,info)
    if len(a)==0:
        print("木材根数:根数差值大于0的错误率统计:", 0)
    else:
        print("木材根数:根数差值大于0的错误率统计:", np.sum(a) / len(a))


    err_less_idx = np.where(idx < 0)[0]
    a = abs(info[err_less_idx][:, 1] - info[err_less_idx][:, 2]) / (info[err_less_idx][:, 2])
    print("木材根数:根数差值小于0的错误率统计:", np.sum(a) / len(a))


if __name__ == '__main__':
    base_dir='/output/result_large50'
    pre_img_dir=os.path.join(base_dir,'images')
    save_dir='../wrong_large50'
    main(os.path.join(base_dir,'normal.csv'),'正常',pre_img_dir,save_dir)
    main(os.path.join(base_dir,'occlude.csv'),'遮挡',pre_img_dir,save_dir)
    main(os.path.join(base_dir,'slope.csv'),'倾斜',pre_img_dir,save_dir)