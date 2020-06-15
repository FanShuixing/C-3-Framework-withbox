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
    #     with open(os.path.join(args.root_dir, args.meta_name + '.csv')) as fr:
    #         file_list = pd.read_csv(fr).values
    #     file_list=[]
    #     with open('equal.txt') as fr:
    #         info=fr.readlines()
    #         for each in info:
    #             file_list.append(each.strip()+'.jpg')
    file_list = os.listdir(args.root_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    test(args, file_list, args.model_path)


def test(args, file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    # 增加csv文件的输出
    writer = csv.writer(open(os.path.join(args.output_dir, '%s.csv' % args.meta_name), 'w+'))
    writer.writerow(['image_name', 'predict_num', 'gt_num', 'boxes_nums'])
    # 增加json的输出
    total_dict = {}
    os.mkdir(os.path.join(args.output_dir, 'gt'))
    os.mkdir(os.path.join(args.output_dir, 'pred'))
    save_img_dir = os.path.join(args.output_dir, 'images')
    os.mkdir(save_img_dir)
    with open('predict_counting.csv', 'w+') as fr:
        writer = csv.writer(fr)
        writer.writerow(['ID', "Detections"])
        for filename in tqdm(file_list):
            info_dict = {}

            name_no_suffix = filename.replace('.jpg', '')
            imgname = os.path.join(args.root_dir, filename)

            img = Image.open(imgname)
            img = img.resize((args.image_shape[1], args.image_shape[0]))

            if img.mode == 'L':
                img = img.convert('RGB')

            img = img_transform(img)

            with torch.no_grad():
                img = Variable(img[None, :, :, :]).cuda()
                pred_map, pred_wh, pred_offset = net.test_forward(img)

            sio.savemat(args.output_dir + '/pred/' + name_no_suffix + '.mat',
                        {'data': pred_map.squeeze().cpu().numpy() / 100.})
            # 处理框
            #             heat = _nms(pred_map / 100.)
            batch = 1
            inds, ys, xs = get_topk(pred_map)
            #             print(xs)
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
            img_show = cv2.imread(imgname)
            ori_h, ori_w, _ = img_show.shape
            #             img_show = cv2.resize(img_show, (768, 576))
            bboxes_json = []
            for i in range(inds.shape[0]):
                tmp = {}
                x0 = int(bboxes[0, i, 0].item())
                y0 = int(bboxes[0, i, 1].item())
                x1 = int(bboxes[0, i, 2].item())
                y1 = int(bboxes[0, i, 3].item())

                # 添加json输出
                tmp['x_min'] = x0 / 768
                tmp['x_max'] = x1 / 768
                tmp['y_min'] = y0 / 576
                tmp['y_max'] = y1 / 576
                tmp['label'] = 'GangJin'
                tmp['confidence'] = 1.0
                bboxes_json.append(tmp)
                #                 ori_x0=int(tmp['x_min']*ori_w)
                #                 ori_y0=int(tmp['y_min']*ori_h)
                #                 ori_x1=int(tmp['x_max']*ori_w)
                #                 ori_y1=int(tmp['y_max']*ori_h)
                ori_x0 = bboxes[0, i, 0].item() / 768 * ori_w
                ori_y0 = bboxes[0, i, 1].item() / 576 * ori_h
                ori_x1 = bboxes[0, i, 2].item() / 768 * ori_w
                ori_y1 = bboxes[0, i, 3].item() / 576 * ori_h
                cv2.rectangle(img_show, (int(ori_x0), int(ori_y0)), (int(ori_x1), int(ori_y1)), (255, 0, 0), 2)

                writer.writerow([name_no_suffix + '.jpg', '%f %f %f %f' % (ori_x0, ori_y0, ori_x1, ori_y1)])
            cv2.imwrite('%s/%s.jpg' % (save_img_dir, name_no_suffix), img_show)

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
                    save_img_dir + '/' + name_no_suffix + '_pred_' + str(round(pred)) + '_gt_' + str(
                        round(gt)) + '_box_' + str(K) + '.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)
            else:
                plt.savefig(
                    save_img_dir + '/' + name_no_suffix + '_pred_' + str(round(pred)) + '_box_' + str(K) + '.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)

            plt.close()

    with open(os.path.join(args.output_dir, '%s.json' % args.meta_name.split('_')[0]), 'w+') as fr:
        json.dump(total_dict, fr)



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
    parser.add_argument('--have_gt', default=False)
    parser.add_argument('--image_shape', default=(576, 768), help='the image shape when training')
    parser.add_argument('--meta_name', default='normal_v16')
    args = parser.parse_args()
    main(args)