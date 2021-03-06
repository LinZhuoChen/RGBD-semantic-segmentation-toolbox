import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize_PIL2numpy_depth2xyz(object):
    """Normalize a tensor image with mean and standard deviation,then
    convert depth to xyz
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
        norm (bool): whether 0-255 or 0-1
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), norm=True):
        self.mean = mean
        self.std = std
        self.norm = norm

    def __call__(self, sample):
        img = sample['image']
        mask = sample['seg']
        HHA = sample['HHA']
        depth = sample['depth']

        if 'scale_x' in sample.keys():
            scale_x = sample['scale_x']
            scale_y = sample['scale_y']
            center_x = sample['center_x']
            center_y = sample['center_y']
        else:
            scale_x = 1.
            scale_y = 1.
            center_x = 0.
            center_y = 0.
            
        ## convert PIL to numpy
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        depth = np.array(depth).astype(np.float32)
        depth = depth[np.newaxis, ...]
        HHA = np.array(HHA).astype(np.float32)

        ## convert depth to xyz
        _, h,w = depth.shape
        z = depth
        xx, yy = np.meshgrid(np.array(range(w))+1, np.array(range(h))+1)
        fx_rgb = 5.18857e+02 * scale_x
        fy_rgb = 5.19469e+02 * scale_y
        cx_rgb = w / 2.0
        cy_rgb = h / 2.0
        C = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
        cc_rgb = C[0:2, 2]
        fc_rgb = np.diag(C[0:2, 0:2])
        x = (np.multiply((xx - cc_rgb[0]), z) / fc_rgb[0])
        y = (np.multiply((yy - cc_rgb[1]), z) / fc_rgb[1])
        depth = np.concatenate([x, y, z], axis=0)

        ## Normalize
        if self.norm:
            img /= 255.0
            img -= self.mean
            img /= self.std

            HHA /= 255.0
            HHA -= self.mean
            HHA /= self.std
 
        else:
            img = (img - np.asarray([122.675,116.669,104.008]))[:,:,::-1]

            HHA = (HHA - np.asarray([122.675,116.669,104.008]))[:,:,::-1]

        depth /= 1000.0
        return  {'image':img,
                'depth':depth,
                'seg': mask,
                'HHA': HHA}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    def __call__(self, sample):
        img = sample['image']
        mask = sample['seg']
        HHA = sample['HHA']
        depth = sample['depth']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        # convert 0-40 to 0-39 and 255
        mask = (np.array(mask).astype(np.uint8)[:, :, 0]-1).astype(np.float32)
        HHA = np.array(HHA).astype(np.float32).transpose((2, 0, 1))
        depth = np.array(depth).astype(np.float32)

        # convert numpy to tensor
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        HHA = torch.from_numpy(HHA).float()

        if len(depth.shape) == 3:
            depth = torch.from_numpy(depth).float()
        else:
            depth = torch.from_numpy(np.expand_dims(depth, axis=0)).float()

        return {'image':img,
                'depth':depth,
                'seg': mask,
                'HHA': HHA}

class RandomHorizontalFlip(object):
    """Random horizontal flip augment
    Args:
        None
    """
    def __call__(self, sample):
        img = sample['image']
        mask = sample['seg']
        HHA = sample['HHA']
        depth = sample['depth']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            HHA = HHA.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image':img,
                'depth':depth,
                'seg': mask,
                'HHA': HHA}

class RandomRotate(object):
    """Random horizontal flip augment
    Args:
        degree: angle range for rotation
    """
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['seg']
        HHA = sample['HHA']
        depth = sample['depth']

        rotate_degree = self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        depth = depth.rotate(rotate_degree, Image.BILINEAR)
        HHA = HHA.rotate(rotate_degree, Image.BILINEAR)

        return  {'image':img,
                'depth':depth,
                'seg': mask,
                'HHA': HHA}

class RandomGaussianBlur(object):
    """Random gaussian blur
    Args:
        degree: angle range for rotation
    """
    def __call__(self, sample):
        img = sample['image']
        mask = sample['seg']
        HHA = sample['HHA']
        depth = sample['depth']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return  {'image':img,
                 'depth':depth,
                'seg': mask,
                'HHA': HHA}

class RandomScaleCrop(object):
    """Random scale crop data augmentation
    Args:
        base_size: base size of image
        crop_size: crop size of image
        fill: fill value
    """
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['seg']
        HHA = sample['HHA']
        depth = sample['depth']

        short_size = random.randint(int(self.base_size * 0.8), int(self.base_size * 1.75))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        scale = ow / w
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        HHA = HHA.resize((ow, oh), Image.BILINEAR)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        # pad crop
        if short_size < self.crop_size_h:
            padh = self.crop_size_h - oh if oh < self.crop_size_h else 0
            padw = self.crop_size_w - ow if ow < self.crop_size_w else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            HHA = ImageOps.expand(HHA, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
            depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)
            print("padding... ... ")

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size_w)
        y1 = random.randint(0, h - self.crop_size_h)
        img = img.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
        mask = mask.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
        HHA = HHA.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
        depth = depth.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))

        center_x = -x1
        center_y = -y1

        return  {'image':img,
                'depth':depth,
                'seg': mask,
                'HHA': HHA,
                'scale_x' : scale,
                'scale_y': scale,
                'center_x': center_x,
                 'center_y': center_y
                 }

class FixScaleCrop(object):
    """Fix scale crop data augmentation
    Args:
        crop_size: crop size of image
    """
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['seg']
        HHA = sample['HHA']
        depth = sample['depth']

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        HHA = HHA.resize((ow, oh), Image.BILINEAR)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))

        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        HHA = HHA.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return  {'image':img,
                'depth':depth,
                'seg': mask,
                'HHA': HHA}

class FixedResize(object):
    """Fix scale crop data augmentation
    Args:
        crop_size: crop size of image
    """
    def __init__(self, size):
        self.size = size # size: (h, w)
    def __call__(self, sample):
        img = sample['image']
        mask = sample['seg']
        HHA = sample['HHA']
        depth = sample['depth']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        HHA = HHA.resize(self.size, Image.BILINEAR)
        depth = depth.resize(self.size, Image.BILINEAR)

        return  {'image':img,
                'depth':depth,
                'seg': mask,
                'HHA': HHA}

class FixedResize_image(object):
    """Resize data augmentation (only for image)
    Args:
        size: new size of image
    """
    def __init__(self, size):
        self.size = size # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['seg']
        HHA = sample['HHA']
        depth = sample['depth']

        assert img.size == mask.size


        img = img.resize(self.size, Image.BILINEAR)
        HHA = HHA.resize(self.size, Image.BILINEAR)
        depth = depth.resize(self.size, Image.BILINEAR)

        return  {'image':img,
                'depth':depth,
                'seg': mask,
                'HHA': HHA}

class CenterCrop(object):
    """center crop augmentation (only for image)
    Args:
        size: crop size
    """
    def __init__(self, size):
        self.size = size 

    def __call__(self, sample):
        img = sample['image']
        mask = sample['seg']
        HHA = sample['HHA']
        depth = sample['depth']

        w, h = img.size
        th, tw = self.size

        x = int(round((w - tw) / 2.))
        y = int(round((h - th) / 2.))

        img = img.crop((x, y, x + tw, y + th))
        mask = mask.crop((x, y, x + tw, y + th))
        HHA = HHA.crop((x, y, x + tw, y + th))
        depth = depth.crop((x, y, x + tw, y + th))

        return  {'image':img,
                'depth':depth,
                'seg': mask,
                'HHA': HHA}
