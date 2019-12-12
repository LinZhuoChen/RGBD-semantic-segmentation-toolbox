from PIL import Image
import math, random
import time
from torch.utils import data
import numpy as np 
import torch
import cv2 
from torchvision import transforms

from .custom_transforms import *

def make_dataset_fromlst(listfilename):
    """
    NYUlist format: imagepath seglabelpath depthpath HHApath
    Args:
        listfilename: file path of NYUlist
    """
    images = []
    segs = []
    depths = []
    HHAs = []

    with open(listfilename) as f:
        content = f.readlines()
        for x in content:
            imgname, segname, depthname, HHAname = x.strip().split(' ')
            images += [imgname]
            segs += [segname]
            depths += [depthname]
            HHAs += [HHAname]

        return {'images':images, 'segs':segs, 'HHAs':HHAs, 'depths':depths}

class NYUDataset_crop(data.Dataset):
    """
    NYUDataset with random crop size 360 * 480
    Args:
        list_path: file path of NYUlist
    """
    def __init__(self, list_path, scale, flip, crop, batch_size, colorjitter, norm=False):
        self.list_path = list_path
        self.scale = scale
        self.flip = flip
        self.crop = crop
        self.batch_size = batch_size
        self.colorjitter = colorjitter

        # np.random.seed(int(time.time()))
        self.paths_dict = make_dataset_fromlst(self.list_path)
        self.len = len(self.paths_dict['images'])
        self.datafile = 'nyuv2_dataset_crop.py'
        self.norm = norm
    def __getitem__(self, index):
        img = Image.open(self.paths_dict['images'][index])
        depth = Image.open(self.paths_dict['depths'][index])
        HHA = Image.open(self.paths_dict['HHAs'][index])
        seg = Image.open(self.paths_dict['segs'][index])

        sample = {'image':img,
                  'depth':depth,
                  'seg': seg,
                  'HHA': HHA}
        sample = self.transform_tr(sample)
        sample = self.totensor(sample)

        return sample
    def __len__(self):
        return self.len

    def name(self):
        return 'NYUDataset_crop'

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomGaussianBlur(),
            RandomScaleCrop(base_size=480, crop_size=[360, 480]),
            Normalize_PIL2numpy_depth2xyz(norm=False),
        ])
        return composed_transforms(sample)

    def totensor(self, sample):
        composed_transforms = transforms.Compose([
            ToTensor()])
        return composed_transforms(sample)

class NYUDataset_crop_fast(data.Dataset):
    """
    NYUDataset with random crop size 480 * 640
    Args:
        list_path: file path of NYUlist
    """
    def __init__(self, list_path, scale, flip, crop, batch_size, colorjitter, norm=False):
        self.list_path = list_path
        self.scale = scale
        self.flip = flip
        self.crop = crop
        self.batch_size = batch_size
        self.colorjitter = colorjitter
        self.norm = norm

        self.paths_dict = make_dataset_fromlst(self.list_path)
        self.len = len(self.paths_dict['images'])
        self.datafile = 'nyuv2_dataset_crop.py'

    def __getitem__(self, index):
        img = Image.open(self.paths_dict['images'][index])
        depth = Image.open(self.paths_dict['depths'][index])
        HHA = Image.open(self.paths_dict['HHAs'][index])
        seg = Image.open(self.paths_dict['segs'][index])

        sample = {'image':img,
                  'depth':depth,
                  'seg': seg,
                  'HHA': HHA}

        sample = self.transform_tr(sample)
        sample = self.totensor(sample)

        return sample

    def __len__(self):
        return self.len

    def name(self):
        return 'NYUDataset_crop_fast'

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomGaussianBlur(),
            RandomScaleCrop(base_size=640, crop_size=[480, 640]),
            Normalize_PIL2numpy_depth2xyz(norm=self.norm),
        ])
        return composed_transforms(sample)

    def totensor(self, sample):
        composed_transforms = transforms.Compose([
            ToTensor()])
        return composed_transforms(sample)

class NYUDataset_val_full(data.Dataset):
    """
    NYUDataset for evaluation with full size
    Args:
        list_path: file path of NYUlist
    """
    def __init__(self, list_path, scale, flip, crop, batch_size, norm=False):
        self.list_path = list_path
        self.scale = scale
        self.flip = flip
        self.crop = crop
        self.batch_size = batch_size
        self.colorjitter = False
        self.norm = norm

        self.paths_dict = make_dataset_fromlst(self.list_path)
        self.len = len(self.paths_dict['images'])

    def __getitem__(self, index):
        # self.paths['images'][index]
        img = Image.open(self.paths_dict['images'][index])  # .astype(np.uint8)
        depth = Image.open(self.paths_dict['depths'][index])
        HHA = Image.open(self.paths_dict['HHAs'][index])
        seg = Image.open(self.paths_dict['segs'][index])

        sample = {'image':img,
                  'depth':depth,
                  'seg': seg,
                  'HHA': HHA}

        sample = self.transform_val(sample)
        sample = self.totensor(sample)

        return sample

    def __len__(self):
        return self.len

    def name(self):
        return 'NYUDataset_val_full'

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            Normalize_PIL2numpy_depth2xyz(norm=self.norm)])
        return composed_transforms(sample)

    def totensor(self, sample):
        composed_transforms = transforms.Compose([
            ToTensor()])
        return composed_transforms(sample)

class NYUDataset_val_crop(data.Dataset):
    """
    NYUDataset for evaluation with 360*480 size input
    Args:
        list_path: file path of NYUlist
    """
    def __init__(self, list_path, scale, flip, crop, batch_size, norm=False):
        self.list_path = list_path
        self.scale = scale
        self.flip = flip
        self.crop = crop
        self.batch_size = batch_size
        self.colorjitter = False
        self.norm = norm

        self.paths_dict = make_dataset_fromlst(self.list_path)
        self.len = len(self.paths_dict['images'])

    def __getitem__(self, index):
        # self.paths['images'][index]
        img = Image.open(self.paths_dict['images'][index])  # .astype(np.uint8)
        depth = Image.open(self.paths_dict['depths'][index])
        HHA = Image.open(self.paths_dict['HHAs'][index])
        seg = Image.open(self.paths_dict['segs'][index])

        sample = {'image':img,
                  'depth':depth,
                  'seg': seg,
                  'HHA': HHA}
        sample = self.transform_val(sample)
        sample = self.totensor(sample)

        return sample

    def __len__(self):
        return self.len

    def name(self):
        return 'NYUDataset_val_crop'

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            FixedResize_image(size=[480, 360]),
            Normalize_PIL2numpy_depth2xyz(norm=self.norm)])
        return composed_transforms(sample)

    def totensor(self, sample):
        composed_transforms = transforms.Compose([
            ToTensor()])
        return composed_transforms(sample)