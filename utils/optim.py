import argparse
import re
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys 
import os
import os.path as osp
from tensorboardX import SummaryWriter
from torch.nn import functional as F

def lr_poly_exp(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def lr_poly_epoch(base_lr, iter, max_iter, power):
    return base_lr/2.0
def adjust_learning_rate(optimizer, i_iter, args):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = max(lr_poly_exp(args.learning_rate, i_iter, args.num_steps, args.power), 1e-3)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003