import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name,args=None):
        super(CrowdCounter, self).__init__()

        if model_name == 'AlexNet':
            from .SCC_Model.AlexNet import AlexNet as net
        elif model_name == 'VGG':
            from .SCC_Model.VGG import VGG as net
        elif model_name == 'VGG_DECODER':
            from .SCC_Model.VGG_decoder import VGG_decoder as net
        elif model_name == 'MCNN':
            from .SCC_Model.MCNN import MCNN as net
        elif model_name == 'CSRNet':
            from .SCC_Model.CSRNet import CSRNet as net
        elif model_name == 'Res50':
            from .SCC_Model.Res50 import Res50 as net
        elif model_name == 'Res101':
            from .SCC_Model.Res101 import Res101 as net
        elif model_name == 'Res101_SFCN':
            from .SCC_Model.Res101_SFCN import Res101_SFCN as net

        self.CCN = net()
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        if args!=None:
            self.wh_decay=float(args.wh_decay)
            self.offset_decay=float(args.offset_decay)
            self.pos_decay=float(args.pos_decay)

    @property
    def loss(self):
        return self.loss_mse, self.wh_loss, self.all_loss

    def forward(self, img, gt_map, gt_wh, gt_ind, gt_reg_mask, gt_hm_mask,gt_reg):
        density_map, pre_wh ,pred_reg= self.CCN(img)
        # 最初只返回这个loss
        #self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
        # 修改hm loss
        #print(density_map.shape,density_map.squeeze().shape)[8,1,576,768],[8,576,768]
        self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze(), gt_hm_mask.squeeze())

        # wh_loss
        self.crit_wh = RegL1Loss()
        self.wh_loss = self.crit_wh(pre_wh, gt_ind, gt_wh, gt_reg_mask)
        
        #reg_loss
        self.reg_loss = self.crit_wh(pred_reg, gt_ind, gt_reg, gt_reg_mask)
        self.all_loss = 1 * self.loss_mse + self.wh_decay* self.wh_loss+self.offset_decay*self.reg_loss
        return density_map

    def build_loss(self, density_map, gt_data, gt_mask):
        # loss_mse = self.loss_mse_fn(density_map*gt_mask, gt_data*gt_mask)

        #loss_mse = F.mse_loss(density_map * gt_mask, gt_data * gt_mask, size_average=False)
        # loss_mse = loss_mse / (576*768*8 + 1e-4)
        loss_mse_pos = F.mse_loss(density_map * gt_mask, gt_data * gt_mask, size_average=False)
        nums = (gt_mask == 1).nonzero().shape[0]
        loss_mse_pos = loss_mse_pos / (nums + 1e-4)

        loss_mse_ori = F.mse_loss(density_map, gt_data, size_average=True)
        loss_mse = loss_mse_ori + self.pos_decay* loss_mse_pos

        return loss_mse

    def test_forward(self, img):
        density_map, pred_wh,pred_offset = self.CCN(img)
        return density_map, pred_wh,pred_offset


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, ind, target, mask):
        '''
        output.shape:[16,2,128,128],模型输出output['wh'],也有可能是output['reg']
        mask.shape:[16,128],gt['reg_mask']
        ind.shape:[16,128],gt['ind']
        target.shape:[16,128,2],gt['wh'],也有可能是gt['reg']
        '''

        pred = _transpose_and_gather_feat(output, ind)  # [4,130,2]
        mask = mask.unsqueeze(2).expand_as(pred).float()  # [4,130,2]
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)

        loss = loss / (mask.sum() + 1e-4)
        return loss


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()  # [1,128,128,2]
    feat = feat.view(feat.size(0), -1, feat.size(3))  # [1,128*128,2]
    feat = _gather_feat(feat, ind)
    return feat


def _gather_feat(feat, ind, mask=None):
    '''
    feat.shape:[1,8000,1]
    ind.shape:[1,100]
    '''
    #     print(feat.shape,ind.shape)
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat