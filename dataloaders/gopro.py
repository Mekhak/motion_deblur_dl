from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

class GoPro(Dataset):
    def __init__(self, blur_imgs_dir, sharp_imgs_dir, image_height, image_width, is_for_train=False):
        self.blur_imgs_dir = blur_imgs_dir
        self.sharp_imgs_dir = sharp_imgs_dir
        self.image_height = image_height
        self.image_width = image_width
        self.is_for_train = is_for_train

        assert image_height > 0, 'Image height must be greater than 0'
        assert image_width > 0, 'Image width must be greater than 0'

        self.ids = [splitext(file)[0] for file in listdir(blur_imgs_dir)
                    if not file.startswith('.')]

        logging.basicConfig(level=logging.INFO)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, image_h, image_w):
        pil_img = pil_img.resize((image_w, image_h))

        # pil_img.show()
        img_nd = np.array(pil_img)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        sharp_file = glob(self.sharp_imgs_dir + idx + '.*')
        blur_file = glob(self.blur_imgs_dir + idx + '.*')

        assert len(sharp_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {sharp_file}'
        assert len(blur_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {blur_file}'

        # sharp = Image.open(sharp_file[0]).split()[-1]
        sharp = Image.open(sharp_file[0])
        sharp = sharp.convert("RGB")

        blur = Image.open(blur_file[0])
        blur = blur.convert("RGB")

        assert blur.size == sharp.size, \
            f'Blur image and mask image {idx} should be the same size,' \
            f'but are {blur.size} and {sharp.size}'

        # if self.is_for_train:
        #     # some real-time augmentations
        #     img =
        #     mask =

        blur = self.preprocess(blur, self.image_height, self.image_width)
        sharp = self.preprocess(sharp, self.image_height, self.image_width)

        return {
            'blur': torch.from_numpy(blur).type(torch.FloatTensor),
            'sharp': torch.from_numpy(sharp).type(torch.FloatTensor)
        }

# blur_dir =  "E:\\motion_deblur\\GOPRO_Large\\train_new\\blur\\"
# sharp_dir = "E:\\motion_deblur\\GOPRO_Large\\train_new\\sharp\\"
#
# gopro = GoPro(blur_dir, sharp_dir, 400, 600)
# print(gopro[5]['blur'])
