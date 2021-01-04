import torch
from torch.autograd import Function
from tqdm import tqdm


class DiceCoeff(Function):
    def forward(self, input, target):
        ## TODO - Implement dice coefficient metric
        inter = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target)

        return (2 * inter.float()) / union.float()


    ## In case you want to use DiceCoeff loss for training, you will have to implement backward function as well.
    # def backward(self, grad_output):

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def eval_net(net, loader, device):
    net.eval()
    n_val = len(loader)
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(imgs)

            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()

            pbar.update()

    net.train()
    return tot / n_val
