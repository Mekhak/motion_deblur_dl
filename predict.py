import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.transformer_net import TransformerNet
from models.ResnetDecoder import SimpleNet, Test
from dataloaders.gopro import GoPro


def predict_img(net,
                full_img,
                device,
                img_height=448,
                img_width=448):
    net.eval()

    img = torch.from_numpy(GoPro.preprocess(full_img, img_height, img_width))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        ## TODO - get the prediction mask, and resize it to the shape of initial image.
        ##        Example, if the test image has shape (H, W, 3), then we resize it to (1, 3, 448, 448), get the prediction mask (1, 1, 448, 448),
        ##        and before we will return it, we need to resize it to (H, W). 
        ##        Finnaly, probs will be array of shape (H, W), with values from 0 to 1.
        # out = net(img)
        # probs = torch.sigmoid(out)
        # probs = transforms.ToPILImage(probs)
        # probs = transforms.Resize(probs, full_img.size[1]),
        # probs = transforms.ToTensor(probs)

        res = net(img).squeeze().cpu()
        res = res.numpy()
        res = res.transpose((1, 2, 0))
        # if max(res) < 1:
        res *= 255
        print(res.shape)
        res = Image.fromarray(np.uint8(res))

    return res


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
    parser.add_argument('-hh', '--height', dest='height', type=int, default=480,
                        help='Height of images')
    parser.add_argument('-ww', '--width', dest='width', type=int, default=640,
                        help='Width of images')

    return parser.parse_args()


# def mask_to_image(mask):
#     return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    # net = TransformerNet()
    net = SimpleNet()
    # net = Test()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    ## TODO - load trained weights
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    if not os.path.exists(args.output):
        os.mkdir(args.output)
        logging.info('Created directory')

    imgs = glob(os.path.join(args.input, '*'))

    print(imgs)
    i = 0
    for pth in imgs:
        print(pth)
        name = pth.split('/')[-1].split('.')[0]
        logging.info("\nPredicting image {} ...".format(name))

        img = Image.open(pth)
        img_resized = img.resize((args.width, args.height))
        img_resized.save(args.output + "\\" + str(i) + "_blur.jpg")

        mask = predict_img(net=net,
                           full_img=img,
                           img_height=args.height,
                           img_width=args.width,
                           device=device)
        # mask = mask_to_image(mask)
        mask.show()

        mask.save(args.output + "\\" + str(i) +'_res.jpg')

        i += 1
