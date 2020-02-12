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
    writer.writerow(['image_name', 'predict_num', 'gt_num'])
    # 增加json的输出
    info_dict = {}
    for filename in file_list:
        print(filename)
        filename_no_ext = re.sub('.jpg', '', filename)
        imgname = os.path.join(args.root_dir, filename_no_ext + '.jpg')
        if args.have_gt:
            denname = args.root_dir + '/val_label/' + filename_no_ext + '.csv'
            den = pd.read_csv(denname, sep=',', header=None).values
            den = den.astype(np.float32, copy=False)
            gt = np.sum(den)
            sio.savemat(exp_name + '/gt/' + filename_no_ext + '.mat', {'data': den})

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)

        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = net.test_forward(img)

        sio.savemat(exp_name + '/pred/' + filename_no_ext + '.mat', {'data': pred_map.squeeze().cpu().numpy() / 100.})

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
            plt.savefig(exp_name + '/' + filename_no_ext + '_gt_' + str(round(gt)) + '.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)

            plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

        pred_frame = plt.gca()
        #         plt.imshow(img)
        #         tmp=cv2.imread(imgname)
        tmp = io.imread(imgname)
        plt.imshow(tmp)
        #         plt.imshow(pred_map, 'jet')
        plt.imshow(pred_map, alpha=0.75)
        print(pred_map.shape)
        io.imsave('/output/pred.jpg', pred_map * 255)
        cv2.imwrite('/output/pred_cv2.jpg', pred_map * 255)
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
            plt.savefig(exp_name + '/' + filename_no_ext + '_diff.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)

            plt.close()

            writer.writerow([imgname, round(pred), round(gt)])
            info_dict[filename_no_ext] = {'pred': str(round(pred)), 'gt': str(round(gt))}
        else:
            writer.writerow([imgname, round(pred)])
            info_dict[filename_no_ext] = {'pred': str(round(pred))}
        # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})
    with open(os.path.join(args.output_dir, 'final_json.json'), 'w+') as fr:
        json.dump(info_dict, fr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default='/output/a', help='root dir')
    parser.add_argument("--model_path",
                        default='/output/tf_dir/02-11_08-14_SHHB_Res101_SFCN_1e-05/all_ep_112_mae_1.3_mse_2.6.pth',
                        help='model path for predict')
    parser.add_argument('--output_dir', default='/output/tf_dir', help='save output')
    parser.add_argument('--have_gt', default=False)
    args = parser.parse_args()
    main(args)