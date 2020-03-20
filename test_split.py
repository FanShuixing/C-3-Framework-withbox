from matplotlib import pyplot as plt
import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps
import re, csv, json
import argparse
import cv2
from skimage import io
from tqdm import tqdm

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

'''
将数据集按照正常，倾斜，遮挡划分来分布进行预测
'''

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
restore = standard_transforms.Compose([
    own_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])
pil_to_tensor = standard_transforms.ToTensor()


def main(args,save_out_img):
    os.makedirs(save_out_img)
    with open(os.path.join(args.root_dir, 'val_crowd_small100.csv')) as fr:
        file_list = pd.read_csv(fr).values
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    test(args, file_list, args.model_path,save_out_img)


def test(args, file_list, model_path,save_out_img):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    f1 = plt.figure(1)

    gts = []
    preds = []
    # 增加csv文件的输出
    writer_occlude = csv.writer(open(os.path.join(args.output_dir, 'occlude.csv'), 'w+'))
    writer_normal = csv.writer(open(os.path.join(args.output_dir, 'normal.csv'), 'w+'))
    writer_slope = csv.writer(open(os.path.join(args.output_dir, 'slope.csv'), 'w+'))
    writer_occlude.writerow(['image_name', 'predict_num', 'gt_num'])
    writer_normal.writerow(['image_name', 'predict_num', 'gt_num'])
    writer_slope.writerow(['image_name', 'predict_num', 'gt_num'])

    # 增加json的输出
    info_dict = {}
    for filename in tqdm(file_list):
        name_no_suffix=filename[0].split('/')[-1].replace('.npy','')
        imgname = os.path.join(args.root_dir,filename[1])
        if args.have_gt:
            denname = os.path.join(args.root_dir,filename[0])
            den=np.load(denname)
            den = den.astype(np.float32, copy=False)
            gt = np.sum(den)

        img = Image.open(imgname)
        img=img.resize((args.image_shape[1],args.image_shape[0]))

        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)

        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = net.test_forward(img)


        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        pred = np.sum(pred_map) / 100.0
        pred_map = pred_map / np.max(pred_map + 1e-20)

        if args.have_gt:
            den = den / np.max(den + 1e-20)
            den_frame = plt.gca()
            plt.imshow(den, 'jet')
            den_frame.axes.get_yaxis().set_visible(False)
            den_frame.axes.get_xaxis().set_visible(False)
            den_frame.spines['top'].set_visible(False)
            den_frame.spines['bottom'].set_visible(False)
            den_frame.spines['left'].set_visible(False)
            den_frame.spines['right'].set_visible(False)
            plt.savefig(save_out_img + '/' + name_no_suffix + '_gt_' + str(round(gt)) + '.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)

            plt.close()


        pred_frame = plt.gca()
        tmp = io.imread(imgname)
        tmp=cv2.resize(tmp,(args.image_shape[1],args.image_shape[0]))
        plt.imshow(tmp)
        plt.imshow(pred_map, alpha=0.75)
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        plt.savefig(save_out_img + '/' + name_no_suffix + '_pred_' + str(round(pred)) + '.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()

        if args.have_gt:
            diff = den - pred_map

            diff_frame = plt.gca()
            plt.imshow(diff, 'jet')
            plt.colorbar()
            diff_frame.axes.get_yaxis().set_visible(False)
            diff_frame.axes.get_xaxis().set_visible(False)
            diff_frame.spines['top'].set_visible(False)
            diff_frame.spines['bottom'].set_visible(False)
            diff_frame.spines['left'].set_visible(False)
            diff_frame.spines['right'].set_visible(False)
            plt.savefig(save_out_img + '/' + name_no_suffix + '_diff.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)

            plt.close()
            if filename[-1]=='正常':
                writer_normal.writerow([imgname, round(pred), round(gt)])
            elif filename[-1]=='倾斜':
                writer_slope.writerow([imgname, round(pred), round(gt)])
            elif filename[-1]=='遮挡':
                writer_occlude.writerow([imgname, round(pred), round(gt)])
        else:
            writer.writerow([imgname, round(pred)])
    with open(os.path.join(args.output_dir, 'final_json.json'), 'w+') as fr:
        json.dump(info_dict, fr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default='/input0/normal', help='root dir')
    parser.add_argument("--model_path",
                        default='/output/tf_dir/03-19_09-30_SHHB_Res101_1e-05/all_ep_85_mae_0.9_mse_2.6.pth',
                        help='model path for predict')
    parser.add_argument('--output_dir', default='/output/result_large50', help='save output')
    parser.add_argument('--have_gt', default=True)
    parser.add_argument('--image_shape', default=(576,768),help='the image shape when training')
    args = parser.parse_args()
    main(args,save_out_img='/output/result_large50/images')