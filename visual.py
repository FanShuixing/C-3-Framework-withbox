import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import os

'''
测试输出的csv文件是否有问题，可视化框
'''
def main():
    os.mkdir('debug')
    with open('predict_counting.csv') as fr:
        info=pd.read_csv(fr).values
        print(info.shape)
    name_list=list(set(info[:,0]))
    for name in tqdm(name_list):
        idx=np.where(info[:,0]==name)[0]
        img=cv2.imread('/input0/test_dataset/%s'%name)
        for each in idx:
            box=info[each,1].split(' ')
#             print(box)
#             print(int(box[0]))
            box=[int(float(each)) for each in box]
#             print(box)
#             print(box[0],box[1],box[2],box[3])
            cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
        cv2.imwrite('debug/%s_%s_counting.jpg'%(name.replace('.jpg',''),idx.shape[0]),img)
        
main()
        