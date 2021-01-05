import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils.eval_utils import eval_net
from models.transformer_net import TransformerNet

from torch.utils.tensorboard import SummaryWriter
from dataloaders.gopro import GoPro
from torch.utils.data import DataLoader

# train_dir_img = 'data/MattingHuman/train/imgs/'
# train_dir_mask = 'data/MattingHuman/train/masks/'
# val_dir_img = 'data/MattingHuman/val/imgs/'
# val_dir_mask = 'data/MattingHuman/val/masks/'

train_dir_img = 'E:\\motion_deblur\\GOPRO_Large\\train_new\\blur\\'
train_dir_mask = 'E:\\motion_deblur\\GOPRO_Large\\train_new\\sharp\\'
val_dir_img = 'E:\\motion_deblur\\GOPRO_Large\\val_new\\blur\\'
val_dir_mask = 'E:\\motion_deblur\\GOPRO_Large\\val_new\\sharp\\'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_height=448,
              img_width=448):

    train_dataset = GoPro(train_dir_img, train_dir_mask, img_height, img_width, True)
    val_dataset = GoPro(val_dir_img, val_dir_mask, img_height, img_width, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    writer = SummaryWriter(log_dir='summary', comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images height:   {img_height}
        Images width:    {img_width}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # TODO - Define two losses, binary cross entropy and mean squared error (use pytorch built in functions)
    # bce = nn.BCELoss()
    mae = nn.L1Loss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                blur_imgs = batch['blur']
                sharp_imgs = batch['sharp']

                blur_imgs = blur_imgs.to(device=device, dtype=torch.float32)
                sharp_imgs = sharp_imgs.to(device=device, dtype=torch.float32)

                sharp_pred = net(blur_imgs)
                # masks_pred = torch.sigmoid(masks_pred)

                # loss_bce = bce(masks_pred, true_masks)
                loss = mae(sharp_imgs, sharp_pred)

                # TODO - tune the hyperparameters w_1 and w_2
                # w_1 = 0.6
                # w_2 = 0.4
                # loss = w_1*loss_bce + w_2*loss_mse

                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)
                # writer.add_scalar('Loss/train/bce', loss_bce.item(), global_step)
                # writer.add_scalar('Loss/train/mse', loss_mse.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()

                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(blur_imgs.shape[0])
                global_step += 1
                if global_step % 2 == 0: #(n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val_score, val_loss = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation PSNR: {}, Validation Loss: {}'
                                 .format(val_score, val_loss))

                    # writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('blur_images', blur_imgs, global_step)
                    writer.add_images('sharp/true', sharp_imgs, global_step)
                    writer.add_images('sharp/pred', sharp_pred, global_step)

        logging.info('Epoch MAE loss: {}'.format(epoch_loss / (n_train // batch_size)))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train a simple network on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
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

    # n_classes is the number of probabilities you want to get per pixel
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = TransformerNet()

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
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
