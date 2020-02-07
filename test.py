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

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = '../SHHB_results'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

if not os.path.exists(exp_name + '/pred'):
    os.mkdir(exp_name + '/pred')

if not os.path.exists(exp_name + '/gt'):
    os.mkdir(exp_name + '/gt')

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


def main(args):
    with open(os.path.join(args.root_dir, 'val.txt')) as fr:
        file_list = [each.strip() for each in fr.readlines()]
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    test(args, file_list, args.model_path)


def test(args, file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    f1 = plt.figure(1)

    gts = []
    preds = []
    # 增加csv文件的输出
    writer = csv.writer(open(os.path.join(args.output_dir, 'final.csv'), 'w+'))
    writer.writerow(['image_name', 'gt_num', 'predict_num'])
    # 增加json的输出
    info_dict = {}
    for filename in file_list:
        print(filename)
        filename_no_ext = re.sub('.csv', '', filename)
        imgname = args.root_dir + '/image/' + filename_no_ext + '.jpg'
        denname = args.root_dir + '/val_label/' + filename_no_ext + '.csv'

        den = pd.read_csv(denname, sep=',', header=None).values
        den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)

        gt = np.sum(den)
        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = net.test_forward(img)

        sio.savemat(exp_name + '/pred/' + filename_no_ext + '.mat', {'data': pred_map.squeeze().cpu().numpy() / 100.})
        sio.savemat(exp_name + '/gt/' + filename_no_ext + '.mat', {'data': den})

        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        pred = np.sum(pred_map) / 100.0
        pred_map = pred_map / np.max(pred_map + 1e-20)

        den = den / np.max(den + 1e-20)

        den_frame = plt.gca()
        plt.imshow(den, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False)
        den_frame.spines['bottom'].set_visible(False)
        den_frame.spines['left'].set_visible(False)
        den_frame.spines['right'].set_visible(False)
        plt.savefig(exp_name + '/' + filename_no_ext + '_gt_' + str(round(gt)) + '.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        plt.savefig(exp_name + '/' + filename_no_ext + '_pred_' + str(round(pred)) + '.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})

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
        plt.savefig(exp_name + '/' + filename_no_ext + '_diff.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()
        writer.writerow([imgname, round(gt), round(pred)])

        info_dict[filename_no_ext] = {'gt': str(round(gt)), 'pred': str(round(pred))}
        # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})
    with open(os.path.join(args.output_dir, 'final_json.json'), 'w+') as fr:
        json.dump(info_dict, fr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default='/input0', help='root dir')
    parser.add_argument("--model_path", default='/output/all_ep_103_mae_1.1_mse_2.3.pth', help='model path for predict')
    parser.add_argument('--output_dir', default='/output/tf_dir', help='save output')
    args = parser.parse_args()
    main(args)
