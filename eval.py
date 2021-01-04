import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils.eval_utils import eval_net
from models.simple_net import SimpleNet

from torch.utils.tensorboard import SummaryWriter
from dataloaders.matting_human import MattingHuman
from torch.utils.data import DataLoader

test_dir_img = 'data/MattingHuman/test/imgs/'
test_dir_mask = 'data/MattingHuman/test/masks/'
dir_checkpoint = 'checkpoints/'

def best_ckpt(net,
              device,
              batch_size=1,
              img_height=448,
              img_width=448):

    test_dataset = MattingHuman(test_dir_img, test_dir_mask, img_height, img_width, False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    n_test = len(test_dataset)

    ckpts = glob(os.path.join('checkpoints/', '*'))
    ckpts.sort(key=lambda item: (len(item), item))

    n_ckpt = len(ckpts)

    logging.info(f'''Starting best checkpoint selection:
        Batch size:      {batch_size}
        Test size:       {n_test}
        # of ckpts:      {n_ckpt}
        Device:          {device.type}
        Images height:   {img_height}
        Images width:    {img_width}
    ''')

    for ckpt in ckpts:
        name = ckpt.split('/')[-1]
        ## TODO - load trained weights from checkpoint and evaluate model on test set.
        net.load_state_dict(torch.load(ckpt), strict=False)
        score = eval_net(net, test_loader, device)
        logging.info('Dice Coeff for {} : {}'.format(name, score))


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints on test set.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-hh', '--height', dest='height', type=int, default=448,
                        help='Height of training images')
    parser.add_argument('-ww', '--width', dest='width', type=int, default=448,
                        help='Width of training images')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = SimpleNet(n_classes=1)
    net.to(device=device)

    try:
        best_ckpt(net=net,
                  batch_size=args.batchsize,
                  device=device,
                  img_height=args.height,
                  img_width=args.width)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)