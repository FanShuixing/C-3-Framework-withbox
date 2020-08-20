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
import torch.nn as nn
from datasets.mat_to_npy import get_density

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
    with open(os.path.join(args.root_dir, args.meta_name + '.csv')) as fr:
        file_list = pd.read_csv(fr).values
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    test(args, file_list, args.model_path)


def test(args, file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    # 增加csv文件的输出
    writer = csv.writer(open(os.path.join(args.output_dir, '%s.csv' % args.meta_name.split('_')[1]), 'w+'))
    writer.writerow(['image_name', 'predict_num', 'gt_num', 'boxes_nums'])
    # 增加json的输出
    total_dict = {}
    os.mkdir(os.path.join(args.output_dir, 'gt'))
    os.mkdir(os.path.join(args.output_dir, 'pred'))
    save_img_dir = os.path.join(args.output_dir, 'images')
    os.mkdir(save_img_dir)
    for filename in tqdm(file_list):
        info_dict = {}

        name_no_suffix = filename[0].split('/')[-1].replace('.json', '')
        imgname = os.path.join(args.root_dir, filename[1])
        if args.have_gt:
            denname = os.path.join(args.root_dir, filename[0])
            den,_ = get_density(os.path.join(args.root_dir, filename[1]), os.path.join(args.root_dir, filename[0]), w=768,h=576)
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
            pred_map, pred_wh,pred_offset = net.test_forward(img)

        sio.savemat(args.output_dir + '/pred/' + name_no_suffix + '.mat',
                    {'data': pred_map.squeeze().cpu().numpy() / 100.})
        # 处理框
        heat = _nms(pred_map / 100.)
        batch = 1
        #         scores, inds, clses, ys, xs = _topk(heat, K=K)
        inds, ys, xs = get_topk(pred_map)
        K = inds.shape[1]

        wh = _transpose_and_gather_feat(pred_wh, torch.from_numpy(inds).cuda())

        # 偏移
        offset = _transpose_and_gather_feat(pred_offset, torch.from_numpy(inds).cuda()).cpu().numpy()
        xs = np.add(xs, offset[:, :, 0:1])
        ys = np.add(ys, offset[:, :, 0:1])

        # 用tensor计算这一步的时候会算错，计算错误
        wh = wh.cpu().numpy()
        bboxes = np.concatenate([xs - wh[..., 0:1] / 2,
                                 ys - wh[..., 1:2] / 2,
                                 xs + wh[..., 0:1] / 2,
                                 ys + wh[..., 1:2] / 2], axis=2)
        img_show = cv2.imread(imgname)[:, :, ::-1]
        img_show = cv2.resize(img_show, (768, 576))
        img_show2=cv2.imread(imgname)[:,:,::-1]
        img_show2=cv2.resize(img_show2,(768,576))
        bboxes_json = []
        for i in range(K):
            tmp = {}
            x0 = int(bboxes[0, i, 0].item())
            y0 = int(bboxes[0, i, 1].item())
            x1 = int(bboxes[0, i, 2].item())
            y1 = int(bboxes[0, i, 3].item())
            cv2.rectangle(img_show, (x0, y0), (x1, y1), (255, 0, 0), 2)
            #             cv2.rectangle(img_show, (xs[0, i, 0], ys[0, i, 0]), (xs[0, i, 0] + 5, ys[0, i, 0] + 5), (255, 0, 0), 2)
            # 添加json输出
            tmp['x_min'] = x0 / 768
            tmp['x_max'] = x1 / 768
            tmp['y_min'] = y0 / 576
            tmp['y_max'] = y1 / 576
            tmp['label'] = 'GangJin'
            tmp['confidence'] = 1.0
            bboxes_json.append(tmp)
        # 给图像画上gt的框
        if args.have_gt:
            with open(os.path.join(args.root_dir, 'mask_labels', name_no_suffix + '.json')) as fr:
                gt_info = json.load(fr)
                for each_box in gt_info['bboxes']:
                    x_min = each_box['x_min'] * args.image_shape[1]
                    y_min = each_box['y_min'] * args.image_shape[0]
                    x_max = each_box['x_max'] * args.image_shape[1]
                    y_max = each_box['y_max'] * args.image_shape[0]

                    x_center = int((x_max - x_min) / 2 + x_min)
                    y_center = int((y_max - y_min) / 2 + y_min)
#                     cv2.rectangle(img_show, (x_center, y_center), (x_center + 5, y_center + 5), (0, 0, 255), 2)
        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        pred = np.sum(pred_map) / 100.0
        pred_map = pred_map / np.max(pred_map + 1e-20)

        pred_frame = plt.gca()

#         plt.imshow(img_show)
        img_show=cv2.putText(img_show, 'GT:'+str(round(gt)), (25, 25), cv2.FONT_HERSHEY_SIMPLEX,\
                                1, (255, 255, 255), 3, cv2.LINE_AA, False)
# img=cv2.resize(img,(512,612))
        img_show=cv2.putText(img_show, 'Pred:'+str(round(K)), (25, 60), cv2.FONT_HERSHEY_SIMPLEX,\
                                1, (255, 255, 255), 3, cv2.LINE_AA, False)
        plt.imshow(img_show)
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        if args.have_gt:
            plt.savefig(
                save_img_dir + '/' + name_no_suffix + '_pred_' + str(round(pred)) + '_gt_' + str(
                    round(gt)) + '_box_' + str(K) + '.png', \
                bbox_inches='tight', pad_inches=0, dpi=150)
        else:
            plt.savefig(save_img_dir + '/' + name_no_suffix + '_pred_' + str(round(pred)) + '_box_' + str(K) + '.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()
        
        #
        pred_frame = plt.gca()
        plt.imshow(img_show2)
        plt.imshow(pred_map, alpha=0.5)
        
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        plt.savefig(save_img_dir + '/' + name_no_suffix + '_pred_' + str(round(pred)) + '_box_' + str(K) +"dmap" +'.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)        
        plt.close()
        

        if args.have_gt:
            writer.writerow([imgname, round(pred), round(gt), K])
            info_dict['image_height'] = 768
            info_dict['image_width'] = 576
            info_dict['num_box'] = len(bboxes_json)
            info_dict['bboxes'] = bboxes_json
            total_dict[name_no_suffix] = info_dict
        else:
            writer.writerow([imgname, round(pred), K])
            info_dict[name_no_suffix] = {'pred': str(round(pred))}
    with open(os.path.join(args.output_dir, '%s.json' % args.meta_name.split('_')[0]), 'w+') as fr:
        json.dump(total_dict, fr)


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
                        default='/output/tf_dir/04-03_10-19_SHHB_Res101_1e-05/all_ep_61_mae_0.9_mse_1.8.pth',
                        help='model path for predict')
    parser.add_argument('--output_dir', default='../result_16_normal', help='save output')
    parser.add_argument('--have_gt', default=True)
    parser.add_argument('--image_shape', default=(576, 768), help='the image shape when training')
    parser.add_argument('--meta_name', default='normal_v16')
    args = parser.parse_args()
    main(args)