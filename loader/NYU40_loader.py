import collections
import torch
import numpy as np
import scipy.misc as m
from torch.utils import data
import random
import numbers

random.seed(19)

class NYU40Loader(data.Dataset):
    def __init__(self, root, split="test", is_transform=False, img_size=512):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 41
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = collections.defaultdict(list)

        for split in [self.split]:
            file_list = tuple(open(root + split + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/input/' + img_name + '.jpg'
        lbl_path = self.root + '/labels_40/' + img_name + '.png'

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform_test(img, lbl)

        return img, lbl


    def transform_test(self, img, lbl):
        img = img.astype(float)
        hei,wid,_=img.shape
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(float) / 255.0##<----------------------
        lbl[lbl == 255] = 0
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)

        co_transform = Compose([
            CenterCrop((228, 304)),
            RandomHorizontalFlip(),
        ])
        img, lbl = co_transform(img, lbl)
        img = img.transpose(2, 0, 1)

        lbl = m.imresize(lbl, (480, 640), 'nearest', mode='F')
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input,target_label):
        for i,t in enumerate(self.co_transforms):
            input,target_label = t(input,target_label)
        return input,target_label

class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target_label):
        h, w, _ = inputs.shape
        th, tw = self.size
        x = int(round((w - tw) / 2.))
        y = int(round((h - th) / 2.))

        inputs = inputs[y : y + th, x : x + tw]
        target_label = target_label[y : y + th, x : x + tw]

        return inputs,target_label

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, inputs,target_label):
        if random.random() < 0.5:
            inputs = np.fliplr(inputs).copy()
            target_label = np.fliplr(target_label).copy()
        return inputs,target_label

if __name__ == '__main__':
    local_path = 'NYU2/data'
    dst = NYU40Loader(local_path, is_transform=True)
