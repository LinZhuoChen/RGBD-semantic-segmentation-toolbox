import argparse
## nyud data
DATASET = 'NYUD'
BATCH_SIZE = 8
DATA_DIRECTORY = 'dataset/voc_12aug'
DATA_LIST_PATH = './dataset/list/nyud/train_nyud_2.txt'
DATA_VAL_LIST_PATH = './dataset/list/nyud/test_nyud_2.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 5e-3
MOMENTUM = 0.9
NUM_CLASSES = 40
NUM_STEPS = 40000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/resnet101.pth'
RESTORE_FROM2 = './dataset/resnet50_.pth'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 5e-4
GPU_ID = "0,1"
EPOCH = 500
NAME = 'resnet101'
DEFORMABLE = True
MODEL_FILE= './networks/baseline.py'
MODULE_FILE= './modules/modulated_dcn.py'
CONFIG_FILE= 'config.py'
TRAIN_FILE= 'train.py'
DATASET_FILE='./dataset/datasets.py'
TRANSFORM_FILE='./dataset/custom_transforms.py'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--name", type=str, default=NAME,
                        help="The name of this exp.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--model_file", type=str, default=MODEL_FILE,
                        help="model file in this exp.")
    parser.add_argument("--module_file", type=str, default=MODULE_FILE,
                        help="module file in this exp.")
    parser.add_argument("--train_file", type=str, default=TRAIN_FILE,
                        help="train file in this exp.")
    parser.add_argument("--dataset_file", type=str, default=DATASET_FILE,
                        help="train file in this exp.")
    parser.add_argument("--transform_file", type=str, default=TRANSFORM_FILE,
                        help="train file in this exp.")
    parser.add_argument("--config_file", type=str, default=CONFIG_FILE,
                        help="config file in this exp.")    
    parser.add_argument("--data-val-list", type=str, default=DATA_VAL_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true", default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--colorjitter", action="store_true", default=True,
                        help="Whether to colorjitter.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-crop", action="store_true", default=True,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from2", type=str, default=RESTORE_FROM2,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--epoch", type=int, default=EPOCH,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default=GPU_ID,
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="choose the number of recurrence.")
    parser.add_argument("--ft", type=bool, default=False,
                        help="fine-tune the model with large input size.")

    parser.add_argument("--ohem", type=str2bool, default='False',
                        help="use hard negative mining")
    parser.add_argument("--ohem-thres", type=float, default=0.6,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--ohem-keep", type=int, default=200000,
                        help="choose the samples with correct probability underthe threshold.")

    parser.add_argument("--model-config", type=int, default=0,
                        help="choose the samples with correct probability underthe threshold.")

    return parser.parse_args()
