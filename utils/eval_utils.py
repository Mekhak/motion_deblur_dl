import torch
from tqdm import tqdm
import torch.functional as F
import torch.nn as nn

class Psnr:

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))


def eval_net(net, loader, device):
    net.eval()

    n_val = len(loader)
    tot = 0
    tot_loss = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            blur_imgs, sharp_imgs = batch['blur'], batch['sharp']
            blur_imgs = blur_imgs.to(device=device, dtype=torch.float32)
            sharp_imgs = sharp_imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                preds = net(blur_imgs)

            # pred = torch.sigmoid(mask_pred)
            # pred = (pred > 0.5).float()
            tot += Psnr()(sharp_imgs, preds)
            # tot_loss += nn.L1Loss()(sharp_imgs, preds)
            tot_loss += nn.MSELoss()(sharp_imgs, preds)

            # pbar.update()

    net.train()

    # print()
    # print("n_val: ", n_val)
    # print("batch_size: ", batch_size)
    # print("tot_loss: ", tot_loss)
    # print("n_val // batch_size: ", n_val // batch_size)
    # print("tot_loss / (n_val // batch_size): ", tot_loss / (n_val // batch_size))
    # print()

    return tot / n_val, tot_loss / n_val
