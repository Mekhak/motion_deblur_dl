import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.simple_net import SimpleNet
from dataloaders.matting_human import MattingHuman


def predict_img(net,
                full_img,
                device,
                img_height=448,
                img_width=448):
    net.eval()

    img = torch.from_numpy(MattingHuman.preprocess(full_img, img_height, img_width))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        ## TODO - get the prediction mask, and resize it to the shape of initial image.
        ##        Example, if the test image has shape (H, W, 3), then we resize it to (1, 3, 448, 448), get the prediction mask (1, 1, 448, 448),
        ##        and before we will return it, we need to resize it to (H, W). 
        ##        Finnaly, probs will be array of shape (H, W), with values from 0 to 1.
        out = net(img)
        probs = torch.sigmoid(out)
        probs = transforms.ToPILImage(probs)
        probs = transforms.Resize(probs, full_img.size[1]),
        probs = transforms.ToTensor(probs)

    return probs > 0.5


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i',
                        help='Directory name of input images', required=True)
    parser.add_argument('--output', '-o',
                        help='Derectory where outputs will be saved', required=True)
    parser.add_argument('-hh', '--height', dest='height', type=int, default=448,
                        help='Height of images')
    parser.add_argument('-ww', '--width', dest='width', type=int, default=448,
                        help='Width of images')

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    net = SimpleNet(n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    ## TODO - load trained weights
    logging.info("Model loaded !")

    if not os.path.exists(args.output):
        os.mkdir(args.output)
        logging.info('Created directory')

    imgs = glob(os.path.join(args.input, '*'))

    for pth in imgs:
        name = pth.split('/')[-1].split('.')[0]
        logging.info("\nPredicting image {} ...".format(name))

        img = Image.open(pth)

        mask = predict_img(net=net,
                           full_img=img,
                           img_height=args.height,
                           img_width=args.width,
                           device=device)
        mask = mask_to_image(mask)
        mask.save(args.output+name+'.jpg')
