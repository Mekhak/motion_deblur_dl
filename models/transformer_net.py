import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), bias=False, padding=1)
        self.bn1   = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), bias=False, padding=1)
        self.bn2   = nn.BatchNorm2d(128)

        self.activation = nn.ReLU()

    def forward(self,
                input: torch.tensor) -> torch.Tensor:
        # Shape: (batch_size, 128, W, H)
        X = input

        # Shape: (batch_size, 128, W, H)
        X = self.conv1(X)
        X = self.bn1(X)

        # Shape: (batch_size, 128, W, H)
        X = self.activation(X)

        # Shape: (batch_size, 128, W, H)
        X = self.conv2(X)
        X = self.bn2(X)

        # residual connection
        # Shape: (batch_size, 128, W, H)
        X = X + input

        return X


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        self.activation = nn.ReLU()

        # conv + batchnorm + relu, all the convolutions are 'same' convolutions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(9, 9), padding=4)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, stride=2)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, stride=2)
        self.bn3   = nn.BatchNorm2d(128)

        # residual blocks
        self.num_of_res_blocks = 5

        self.res_blocks = torch.nn.ModuleList([
            ResidualBlock() for _ in range(self.num_of_res_blocks)
        ])

        # upconv blocks
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, bias=False)
        self.bn5   = nn.BatchNorm2d(32)

        self.conv6 = nn.Conv2d(32, 3, kernel_size=(9, 9), stride=1, padding=4, bias=False)
        self.bn6   = nn.BatchNorm2d(3)


    def forward(self,
                input: torch.Tensor) -> torch.Tensor:

        # Shape: (batch_size, 3, 256, 256)
        X = input

        # Shape: (batch_size, 32, 256, 256)
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.activation(X)

        # Shape: (batch_size, 64, 128, 128)
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.activation(X)

        # Shape: (batch_size, 128, 64, 64)
        X = self.conv3(X)
        X = self.bn3(X)
        X = self.activation(X)


        # Each Shape: (bacth_size, 128, 64, 64)
        for res_layer in self.res_blocks:
            X = res_layer(X)

        # Shape: (batch_size, 64, 128, 128)
        X = self.conv4(X)
        X = self.bn4(X)
        X = self.activation(X)

        # Shape: (batch_size, 32, 256, 256)
        X = self.conv5(X)
        X = self.bn5(X)
        X = self.activation(X)

        # Shape: (batch_size, 3, 256, 256)
        X = self.conv6(X)
        X = self.bn6(X)
        X = self.activation(X)

        return X
