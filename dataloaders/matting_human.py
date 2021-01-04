from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

class MattingHuman(Dataset):
    def __init__(self, imgs_dir, masks_dir, image_height, image_width, is_for_train):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.image_height = image_height
        self.image_width = image_width
        self.is_for_train = is_for_train

        assert image_height > 0, 'Image height must be greater than 0'
        assert image_width > 0, 'Image width must be greater than 0'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, image_h, image_w):
        pil_img = pil_img.resize((image_w, image_h))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = Image.open(mask_file[0]).split()[-1]
        img = Image.open(img_file[0])
        img = img.convert("RGB")

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # if self.is_for_train:
        #     # TODO - make some real-time augmentations
        #     img =
        #     mask =

        img = self.preprocess(img, self.image_height, self.image_width)
        mask = self.preprocess(mask, self.image_height, self.image_width)

        if not self.is_for_train:
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
