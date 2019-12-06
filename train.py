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
import random
import timeit
import logging
from tensorboardX import SummaryWriter
from torch.nn import functional as F

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.enabled = True
torch.backends.cudnn.deterministic = True
cudnn.benchmark = True

from networks.baseline import Res_Deeplab
from dataset.datasets import NYUDataset_val_full, NYUDataset_crop_fast
from utils.utils import decode_labels, inv_preprocess, decode_predictions, get_confusion_matrix, get_currect_time
from utils.criterion import CriterionDSN
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.log import Visualizer, Log
from config import *
from utils.optim import adjust_learning_rate

start = timeit.default_timer() 
args = get_arguments()

def main():
    args.time = get_currect_time()

    visualizer = Visualizer(args)
    log = Log(args)
    log.record_sys_param()
    log.record_file()

    """Set GPU Environment"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    trainloader = data.DataLoader(NYUDataset_crop_fast(args.data_list, args.random_scale, args.random_mirror, args.random_crop,
                args.batch_size, args.colorjitter),batch_size=args.batch_size,
                shuffle=True, num_workers=4, pin_memory=True)
    valloader = data.DataLoader(NYUDataset_val_full(args.data_val_list, args.random_scale, args.random_mirror, args.random_crop,
                       1), batch_size=8, shuffle=False, pin_memory=True)

    """Create Network"""
    deeplab = Res_Deeplab(num_classes=args.num_classes)
    print(deeplab)

    """Load pretrained Network"""
    saved_state_dict = torch.load(args.restore_from)
    print(args.restore_from)
    new_params = deeplab.state_dict().copy()
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        # if not i_parts[1]=='layer5':
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

    deeplab.load_state_dict(new_params)

    model = deeplab
    model.cuda()
    model.train()
    model = model.float()
    model = DataParallelModel(model, device_ids=[0, 1])

    criterion = CriterionDSN()
    criterion = DataParallelCriterion(criterion)

    optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate }],
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)

    optimizer.zero_grad()

    i_iter = 0
    args.num_steps = len(trainloader) * args.epoch
    best_iou = 0.0
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

    for epoch in range(args.epoch):
        ## Train one epoch
        model.train()
        for batch in trainloader:
            start = timeit.default_timer()
            i_iter = i_iter + 1
            images = batch['image'].cuda()
            labels = batch['seg'].cuda()
            HHAs = batch['HHA'].cuda()
            depths = batch['depth'].cuda()
            labels = torch.squeeze(labels,1).long()
            if (images.size(0) != args.batch_size):
                break
            optimizer.zero_grad()
            preds = model(images, HHAs, depths)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            if i_iter % 100 == 0:
                visualizer.add_scalar('learning_rate', args.learning_rate, i_iter)
                visualizer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

            current_lr = optimizer.param_groups[0]['lr']
            end = timeit.default_timer()
            log.log_string(
                '====================> epoch=%03d/%d, iter=%05d/%05d, loss=%.3f, %.3fs/iter, %02d:%02d:%02d, lr=%.6f' % (
                    epoch, args.epoch, i_iter, len(trainloader)*args.epoch, loss.data.cpu().numpy(), (end - start),
                    (int((end - start) * (args.num_steps - i_iter)) // 3600),
                    (int((end - start) * (args.num_steps - i_iter)) % 3600 // 60),
                    (int((end - start) * (args.num_steps - i_iter)) % 3600 % 60), current_lr))
        if (epoch+1) % 40 == 0:
            adjust_learning_rate(optimizer, i_iter, args)

        if epoch % 5 == 0:
            model.eval()
            confusion_matrix = np.zeros((args.num_classes, args.num_classes))
            loss_val = 0
            log.log_string("====================> evaluating")
            for batch_val in valloader:
                images_val = batch_val['image'].cuda()
                labels_val = batch_val['seg'].cuda()
                labels_val = torch.squeeze(labels_val,1).long()
                HHAs_val = batch_val['HHA'].cuda()
                depths_val = batch_val['depth'].cuda()

                with torch.no_grad():
                    preds_val = model(images_val, HHAs_val, depths_val)
                    loss_val += criterion(preds_val, labels_val)
                    preds_val = torch.cat([preds_val[i][0] for i in range(len(preds_val))], 0)
                    preds_val = F.upsample(input=preds_val, size=(480, 640), mode='bilinear', align_corners=True)

                    preds_val = np.asarray(np.argmax(preds_val.cpu().numpy(), axis=1), dtype=np.uint8)

                    labels_val = np.asarray(labels_val.cpu().numpy(), dtype=np.int)
                    ignore_index = labels_val != 255

                    labels_val = labels_val[ignore_index]
                    preds_val = preds_val[ignore_index]

                    confusion_matrix += get_confusion_matrix(labels_val, preds_val, args.num_classes)
            loss_val = loss_val / len(valloader)
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)

            IU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IU = IU_array.mean()

            # getConfusionMatrixPlot(confusion_matrix)
            log.log_string('val loss' + ' ' + str(loss_val.cpu().numpy()) + ' ' + 'meanIU' + str(mean_IU) + 'IU_array' + str(IU_array))

            visualizer.add_scalar('val loss', loss_val.cpu().numpy(), epoch)
            visualizer.add_scalar('meanIU', mean_IU, epoch)

            if mean_IU > best_iou:
                best_iou = mean_IU
                log.log_string('save best model ...')
                torch.save(deeplab.state_dict(),
                           osp.join(args.snapshot_dir, 'model', args.dataset + NAME + 'best_iu' + '.pth'))

        if epoch % 5 == 0:
            log.log_string('save model ...')
            torch.save(deeplab.state_dict(),osp.join(args.snapshot_dir,'model', args.dataset+ NAME + str(epoch)+'.pth'))

if __name__ == '__main__':
    main()