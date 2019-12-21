import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import json

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from networks.baseline import Res_Deeplab
from dataset.datasets import NYUDataset_val_full, NYUDataset_val_crop
from collections import OrderedDict
import os 
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage


import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = 'NYUD'
DATA_LIST_PATH = './dataset/list/nyud/test_nyud_2.txt'
# DATA_LIST_PATH = './dataset/list/sunrgbd/test_sunrgbd.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 40
# NUM_CLASSES = 37
INPUT_SIZE = '360,480'
RESTORE_FROM = './model.pth'
from tqdm import tqdm

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="choose the number of recurrence.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--whole", type=bool, default=True,
                        help="use whole input size.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--colorjitter", action="store_true",
                        help="Whether to colorjitter.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-crop", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    return parser.parse_args()

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, image, HHA, Depth, tile_size, classes, flip_evaluation, recurrence):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            hha = HHA[:, :, y1:y2, x1:x2]
            depth = Depth[:, :, y1:y2, x1:x2]

            padded_img = pad_image(img, tile_size)
            padded_hha = pad_image(hha, tile_size)
            padded_depth = pad_image(depth, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            # print("Predicting tile %i" % tile_counter)
            padded_prediction = net(Variable(torch.from_numpy(padded_img), volatile=True).cuda(),
                                    Variable(torch.from_numpy(padded_hha), volatile=True).cuda(),
                                    Variable(torch.from_numpy(padded_depth), volatile=True).cuda())
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1,2,0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs

def predict_whole(net, image, tile_size, recurrence, HHA, depth):
    image = torch.from_numpy(image)
    HHA = torch.from_numpy(HHA)
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    prediction, _, = net(image.cuda(), HHA.cuda(), depth.cuda())
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().data[0].numpy().transpose(1,2,0)
    return prediction

def predict_multiscale(net, image, HHA, depth, tile_size, scales, classes, flip_evaluation, recurrence):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    HHA = HHA.data
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((tile_size[0], tile_size[1], classes))
    for scale in scales:
        scale = float(scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scale_HHA = ndimage.zoom(HHA, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scaled_probs = predict_whole(net, scale_image, tile_size, recurrence, scale_HHA, depth)
        if flip_evaluation == True:
            flip_scaled_probs = predict_whole(net, scale_image[:,:,:,::-1].copy(), tile_size, recurrence, scale_HHA[:,:,:,::-1].copy(), depth)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,::-1,:])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def get_metric(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc, iu[freq > 0]

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # gpu0 = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    h, w = map(int, args.input_size.split(','))
    if args.whole:
        input_size = (480, 640)
    else:
        input_size = (h, w)

    model = Res_Deeplab(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    # testloader = data.DataLoader(NYUDataset_val_crop(args.data_list, args.random_scale, args.random_mirror, args.random_crop,
    #                 1), batch_size=1, shuffle=False, pin_memory=True)
    testloader = data.DataLoader(NYUDataset_val_full(args.data_list, args.random_scale, args.random_mirror, args.random_crop,
                    1), batch_size=1, shuffle=False, pin_memory=True)

    data_list = []
    confusion_matrix = np.zeros((args.num_classes,args.num_classes))
    palette = get_palette(256)

    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('output_depth'):
        os.makedirs('output_depth')
    index = 0
    for batch in tqdm(testloader):

        # image, label, size, name = batch

        image = batch['image'].cuda()

        label = batch['seg'].cuda()
        label = torch.squeeze(label, 1).long()
        size = np.array([label.size(1), label.size(2)])
        input_size = (label.size(1), label.size(2))

        HHA = batch['HHA'].cuda()
        depth = batch['depth'].cuda()
        with torch.no_grad():
            if args.whole:
                # output = predict_multiscale(model, image, HHA, depth, input_size, [0.8, 1.0, 1.25, 1.5, 1.75], args.num_classes, False, args.recurrence)
                output = predict_multiscale(model, image, HHA, depth, input_size, [1.0],
                                            args.num_classes, False, args.recurrence)
            else:
                output = predict_sliding(model, image.cpu().numpy(), HHA.cpu().numpy(), depth.cpu().numpy(), input_size, args.num_classes, True, args.recurrence)
        # padded_prediction = model(Variable(image, volatile=True).cuda())
        # output = interp(padded_prediction).cpu().data[0].numpy().transpose(1,2,0)
        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        output_im = PILImage.fromarray(np.asarray(np.argmax(output, axis=2), dtype=np.uint8))
        output_im.putpalette(palette)
        output_im.save('output/' + str(index) + '.png')
        seg_gt = np.asarray(label[0].cpu().numpy(), dtype=np.int)
    
        ignore_index = seg_gt != 255

        seg_gt = seg_gt[ignore_index]

        seg_pred = seg_pred[ignore_index]
        # show_all(gt, output)
        confusion_matrix += _fast_hist(seg_gt, seg_pred, args.num_classes)
        depth_np = depth.cpu().numpy()[0, -1, ...]
        depth_np = (depth_np - depth_np.min())/depth_np.max()*255

        depth_np = np.asarray(depth_np, dtype=np.uint8)

        output_depth = PILImage.fromarray(depth_np)
        output_depth.save('output_depth/' + str(index) + '.png')
        index = index + 1



    acc, acc_cls, mean_iu, fwavacc, iu = get_metric(confusion_matrix)

    # getConfusionMatrixPlot(confusion_matrix)
    print({'meanIU': mean_iu, 'IU_array': iu, 'acc': acc, 'acc_cls': acc_cls})

if __name__ == '__main__':
    main()
