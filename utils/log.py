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

class Visualizer():
    """
    Visualizer
    Args:
        args:
    """
    def __init__(self, args):
        self.writer = SummaryWriter(osp.join(args.snapshot_dir, args.name + args.time))
        self.args = args
    def add_scalar(self, name, x, y):
        self.writer.add_scalar(name, x, y)
    def add_image(self, name, image, iter):
        self.writer.add_image(name, image, iter)

class Log():
    """
    Log
    Args:
        args:
    """
    def __init__(self, args):
        self.log_path = osp.join(args.snapshot_dir, args.name + args.time)
        self.log = open(osp.join(self.log_path, 'log_train.txt'), 'w')
        self.args = args

    def record_sys_param(self):
        self.log.write(str(self.args) + '\n')

    def record_file(self):
        os.system('cp %s %s'%(self.args.model_file, self.log_path))
        os.system('cp %s %s'%(self.args.train_file, self.log_path))
        os.system('cp %s %s'%(self.args.config_file, self.log_path))
        os.system('cp %s %s' % (self.args.dataset_file, self.log_path))
        os.system('cp %s %s' % (self.args.transform_file, self.log_path))
        os.system('cp %s %s' % (self.args.module_file, self.log_path))

    def log_string(self, out_str):
        self.log.write(out_str + '\n')
        self.log.flush()
        print(out_str)