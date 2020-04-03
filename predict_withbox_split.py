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
predict_withbox.py的另外一个版本，目的是为了将不同类型的数据进行统计
'''
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

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
    writer_occlude = csv.writer(open(os.path.join(args.output_dir, 'occlude.csv'), 'w+'))
    writer_normal = csv.writer(open(os.path.join(args.output_dir, 'normal.csv'), 'w+'))
    writer_slope = csv.writer(open(os.path.join(args.output_dir, 'slope.csv'), 'w+'))
    writer_occlude.writerow(['image_name', 'predict_num', 'gt_num'])
    writer_normal.writerow(['image_name', 'predict_num', 'gt_num'])
    writer_slope.writerow(['image_name', 'predict_num', 'gt_num'])
    # 增加json的输出
    info_dict = {}

    os.mkdir(os.path.join(args.output_dir, 'gt'))
    os.mkdir(os.path.join(args.output_dir, 'pred'))
    save_img_dir = os.path.join(args.output_dir, 'images')
    os.mkdir(save_img_dir)

    for filename in tqdm(file_list):
        name_no_suffix = filename[0].split('/')[-1].replace('.npy', '')
        imgname = os.path.join(args.root_dir, filename[1])
        if args.have_gt:
            denname = os.path.join(args.root_dir, filename[0])
            den = np.load(denname)
            den = den.astype(np.float32, copy=False)
            gt = np.sum(den)
            sio.savemat(args.output_dir + '/gt/' + name_no_suffix + '.mat', {'data': den})
        img = Image.open(imgname)
        img = img.resize((args.image_shape[1], args.image_shape[0]))

        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)

        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map, pred_wh = net.test_forward(img)

        sio.savemat(args.output_dir + '/pred/' + name_no_suffix + '.mat',
                    {'data': pred_map.squeeze().cpu().numpy() / 100.})
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

        pred_frame = plt.gca()

        plt.imshow(img_show)
        plt.imshow(pred_map, alpha=0.5)
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        if args.have_gt:
            plt.savefig(
                save_img_dir + '/' + name_no_suffix + '_pred_' + str(round(pred)) + '_gt_' + str(round(gt)) + '.png', \
                bbox_inches='tight', pad_inches=0, dpi=150)
        else:
            plt.savefig(save_img_dir + '/' + name_no_suffix + '_pred_' + str(round(pred)) + '.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()

        if args.have_gt:
            if filename[-1] == '正常':
                writer_normal.writerow([imgname, round(pred), round(gt)])
            elif filename[-1] == '倾斜':
                writer_slope.writerow([imgname, round(pred), round(gt)])
            elif filename[-1] == '遮挡':
                writer_occlude.writerow([imgname, round(pred), round(gt)])
        else:
            writer.writerow([imgname, round(pred)])
            info_dict[name_no_suffix] = {'pred': str(round(pred))}
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
                        default='/output/tf_dir/04-02_12-11_SHHB_Res101_1e-05/all_ep_61_mae_1.5_mse_2.5.pth',
                        help='model path for predict')
    parser.add_argument('--output_dir', default='../result', help='save output')
    parser.add_argument('--have_gt', default=True)
    parser.add_argument('--image_shape', default=(576, 768), help='the image shape when training')
    args = parser.parse_args()
    main(args)