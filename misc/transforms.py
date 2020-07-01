import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from config import cfg
import torch
from torchvision import transforms


# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, den, hm_mask, bboxes):
        for t in self.transforms:
            img, den, hm_mask, bboxes = t(img, den, hm_mask, bboxes)
        return img, den, hm_mask, bboxes


class RandomColorJitter(object):
    def __call__(self, img, den, hm_mask, bboxes):
        if random.random() < 0.5:
            img = transforms.ColorJitter(brightness=random.uniform(0, 2), contrast=random.uniform(0, 2))(img)

        return img, den, hm_mask, bboxes


class RandomVerticallyFlip(object):
    def __call__(self, img, den, hm_mask, bboxes):
        if random.random() < 0.5:
            w, h = img.size
            box_h = bboxes[:, 3] - bboxes[:, 1]
            bboxes[:, 1] = h - bboxes[:, 1] - box_h
            bboxes[:, 3] = bboxes[:, 1] + box_h
            img, den, hm_mask = img.transpose(Image.FLIP_TOP_BOTTOM), den.transpose(
                Image.FLIP_TOP_BOTTOM), hm_mask.transpose(Image.FLIP_TOP_BOTTOM)
        #         visual_debug(img,bboxes,den,'final')

        return img, den, hm_mask, bboxes


def visual_debug(img, bboxes, den, name):
    # 框也要对应的发生改变

    img_a = np.array(img)
    import cv2
    for each in bboxes:
        x0, y0, x1, y1 = each
        cv2.rectangle(img_a, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    import random
    name_seed = random.random()
    cv2.imwrite('%s/%s.jpg' % (name, name_seed), img_a)
    np.save('%s/%s.npy' % (name, name_seed), den)


class RandomHorizontallyFlip(object):
    def __call__(self, img, den, hm_mask, bboxes):
        if random.random() < 0.5:
            w, h = img.size
            box_w = bboxes[:, 2] - bboxes[:, 0]
            bboxes[:, 0] = w - bboxes[:, 0] - box_w
            bboxes[:, 2] = bboxes[:, 0] + box_w
            return img.transpose(Image.FLIP_LEFT_RIGHT), \
                   den.transpose(Image.FLIP_LEFT_RIGHT), hm_mask.transpose(Image.FLIP_LEFT_RIGHT), bboxes
        return img, den, hm_mask, bboxes


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, hm_mask, bboxes, dst_size=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        if dst_size is None:
            th, tw = self.size
        else:
            th, tw = dst_size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        if random.random() < 0.5:

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            # visual_debug(img,bboxes,den,'a')
            bboxes[:, 0] -= x1
            bboxes[:, 1] -= y1
            bboxes[:, 2] -= x1
            bboxes[:, 3] -= y1

            # 删除x0,y0<0的框
            remove_ixs = np.where(bboxes[:, 0] < 0)[0]
            bboxes = np.delete(bboxes, remove_ixs, axis=0)
            remove_ixs = np.where(bboxes[:, 1] < 0)[0]
            bboxes = np.delete(bboxes, remove_ixs, axis=0)

            # 删除x1>tw的
            remove_ixs = np.where(bboxes[:, 2] > tw)[0]
            bboxes = np.delete(bboxes, remove_ixs, axis=0)
            # 删除y1>th的
            remove_ixs = np.where(bboxes[:, 3] > th)[0]
            bboxes = np.delete(bboxes, remove_ixs, axis=0)

            img = img.crop((x1, y1, x1 + tw, y1 + th))
            den = mask.crop((x1, y1, x1 + tw, y1 + th))
            hm_mask = hm_mask.crop((x1, y1, x1 + tw, y1 + th))
        else:
            img, den, hm_mask = img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th),
                                                                                  Image.NEAREST), hm_mask.resize(
                (tw, th), Image.NEAREST)
            bboxes[:, 0] = bboxes[:, 0] / w * tw
            bboxes[:, 1] = bboxes[:, 1] / h * th
            bboxes[:, 2] = bboxes[:, 2] / w * tw
            bboxes[:, 3] = bboxes[:, 3] / h * th

        # debug
        #         visual_debug(img,bboxes,den,'crop')
        return img, den, hm_mask, bboxes


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]),
                                                                                     Image.NEAREST)


class ScaleDown(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, mask):
        return mask.resize((self.size[1] / cfg.TRAIN.DOWNRATE, self.size[0] / cfg.TRAIN.DOWNRATE), Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor * self.para
        return tensor


class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor == 1:
            return img
        tmp = np.array(img.resize((w // self.factor, h // self.factor), Image.BICUBIC)) * self.factor * self.factor
        img = Image.fromarray(tmp)
        return img
