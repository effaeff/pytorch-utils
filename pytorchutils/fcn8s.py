"""Models for semantic segmentation based on VGG architectures"""

import copy

from pytorchutils.vgg import VGGModel
from pytorchutils.globals import nn


class FCNModel(nn.Module):
    """Actual class for FCN8s model using the VGG architecture as backbone"""
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)

        self.n_channels = self.config.get('n_channels', 3)
        self.output_size = self.config.get('output_size', 0)
        if self.output_size == 0:
            raise ValueError("Error: No output size defined.")

        self.channel_to_vgg = nn.Conv2d(self.n_channels, 3, kernel_size=3, padding=1)

        self.pretrained_net = VGGModel(config)
        self.activation = nn.ReLU(inplace=True)

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.drop7 = nn.Dropout2d()

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.drop7 = nn.Dropout2d()

        self.score_fn = nn.Conv2d(4096, self.output_size, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.output_size, 1)
        self.score_pool3 = nn.Conv2d(256, self.output_size, 1)

        self.upscore = nn.ConvTranspose2d(
            self.output_size,
            self.output_size,
            9,
            stride=8,
            padding=1,
            output_padding=1
        )
        self.upscore4 = nn.ConvTranspose2d(
            self.output_size,
            self.output_size,
            3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.upscore5 = nn.ConvTranspose2d(
            self.output_size,
            self.output_size,
            3,
            stride=2,
            padding=1,
            output_padding=1
        )

    def forward(self, inp):
        """Forward pass"""
        inp = self.channel_to_vgg(inp)
        vgg_out = self.pretrained_net(inp)
        x_5 = vgg_out['x5']  # size = (N, 512, H/32, W/32)
        x_4 = vgg_out['x4']  # size = (N, 512, H/16, W/16)
        x_3 = vgg_out['x3']  # size = (N, 256, H/8,  W/8)

        pred_out = self.activation(self.fc6(x_5))
        pred_out = self.drop6(pred_out)

        pred_out = self.activation(self.fc7(pred_out))
        pred_out = self.drop7(pred_out)

        pred_out = self.score_fn(pred_out)

        upscore5 = self.upscore5(pred_out)

        score4 = self.score_pool4(x_4)
        score4 += upscore5

        upscore4 = self.upscore4(score4)

        score3 = self.score_pool3(x_3)
        score3 += upscore4

        out = self.upscore(score3)
        return out
