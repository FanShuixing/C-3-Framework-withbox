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
from find_key_coor import get_topk, get_wh

'''
从test.py复制得到，增加了框的输出
'''
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
    with open(os.path.join(args.root_dir, 'val_crowd_300.csv')) as fr:
        file_list = pd.read_csv(fr).values
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    test(args, file_list, args.model_path)


def test(args, file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    # 增加csv文件的输出
    writer = csv.writer(open(os.path.join(args.output_dir, 'final.csv'), 'w+'))
    writer.writerow(['image_name', 'predict_num', 'gt_num'])
    # 增加json的输出
    info_dict = {}
    for filename in tqdm(file_list):
        name_no_suffix = filename[0].split('/')[-1].replace('.npy', '')
        imgname = os.path.join(args.root_dir, filename[1])
        if args.have_gt:
            denname = os.path.join(args.root_dir, filename[0])
            den = np.load(denname)
            den = den.astype(np.float32, copy=False)
            gt = np.sum(den)
            sio.savemat(exp_name + '/gt/' + name_no_suffix + '.mat', {'data': den})
        img = Image.open(imgname)
        img = img.resize((args.image_shape[1], args.image_shape[0]))

        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)

        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map, pred_wh = net.test_forward(img)

        sio.savemat(exp_name + '/pred/' + name_no_suffix + '.mat', {'data': pred_map.squeeze().cpu().numpy() / 100.})
        # 处理框
        heat = _nms(pred_map / 100.)
        batch = 1
        #         scores, inds, clses, ys, xs = _topk(heat, K=K)
        inds, ys, xs = get_topk(pred_map)
        K = inds.shape[1]

        wh = _transpose_and_gather_feat(pred_wh, torch.from_numpy(inds).cuda())
        #         wh=get_wh(pred_wh,inds)

        # 用tensor计算这一步的时候会算错，计算错误
        wh = wh.cpu().numpy()
        bboxes = np.concatenate([xs - wh[..., 0:1] / 2,
                                 ys - wh[..., 1:2] / 2,
                                 xs + wh[..., 0:1] / 2,
                                 ys + wh[..., 1:2] / 2], axis=2)
        img_show = cv2.imread('/input1/normal/images/%s.jpg' % name_no_suffix)
        img_show = cv2.resize(img_show, (768, 576))
        for i in range(K):
            x0 = int(bboxes[0, i, 0].item())
            y0 = int(bboxes[0, i, 1].item())
            x1 = int(bboxes[0, i, 2].item())
            y1 = int(bboxes[0, i, 3].item())
            cv2.rectangle(img_show, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.rectangle(img_show, (xs[0, i, 0], ys[0, i, 0]), (xs[0, i, 0] + 5, ys[0, i, 0] + 5), (255, 0, 0), 2)

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
            plt.savefig(exp_name + '/' + name_no_suffix + '_gt_' + str(round(gt)) + '.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)

            plt.close()

        # sio.savemat(exp_name+'/'+filename+'_gt_'+str(int(gt))+'.mat',{'data':den})

        pred_frame = plt.gca()

        plt.imshow(img_show)
        plt.imshow(pred_map, alpha=0.5)
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        plt.savefig(exp_name + '/' + name_no_suffix + '_pred_' + str(round(pred)) + '.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})
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
            plt.savefig(exp_name + '/' + name_no_suffix + '_diff.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)

            plt.close()

            writer.writerow([imgname, round(pred), round(gt)])
            info_dict[name_no_suffix] = {'pred': str(round(pred)), 'gt': str(round(gt))}
        else:
            writer.writerow([imgname, round(pred)])
            info_dict[name_no_suffix] = {'pred': str(round(pred))}
        # sio.savemat(exp_name+'/'+filename+'_diff.mat',{'data':diff})
    with open(os.path.join(args.output_dir, 'final_json.json'), 'w+') as fr:
        json.dump(info_dict, fr)


import torch.nn as nn


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default='/input1/normal', help='root dir')
    parser.add_argument("--model_path",
                        default='/output/all_ep_88_mae_2.1_mse_3.6.pth',
                        help='model path for predict')
    parser.add_argument('--output_dir', default='/output/tf_dir', help='save output')
    parser.add_argument('--have_gt', default=True)
    parser.add_argument('--image_shape', default=(576, 768), help='the image shape when training')
    args = parser.parse_args()
    main(args)