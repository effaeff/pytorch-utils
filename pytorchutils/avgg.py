"""Models for VGG11/13/16/19 architectures for the usage as backbone for FCN models"""

import re
import copy
import numpy as np

from torchvision import models

from pytorchutils.globals import nn, DEVICE
from pytorchutils.layers import (
    Attention,
    LinearAttention,
    Residual,
    PreNorm,
    WSConv2d
)


class AVGGModel(models.vgg.VGG):
    """
    Attentional model using VGG backbone cropped before fully connected layers.
    References:
        - https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    """
    def __init__(self, config, requires_grad=True, rm_fc=True):
        self.config = copy.deepcopy(config)
        arch = self.config.get('arch', 'vgg16')
        pretrained = self.config.get('pretrained', True)
        batch_norm = self.config.get('vgg_bn', False)

        self.convlayer_ranges = {
            'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
            'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
            'vgg16': ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
            'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
        }[arch] if batch_norm else {
            'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
            'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
            'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
            'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
        }[arch]
        self.convlayer_configs = {
            'vgg11': [
                64, 'M',
                128, 'M',
                256, 256, 'M',
                512, 512, 'M',
                512, 512, 'M'
            ],
            'vgg13': [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 'M',
                512, 512, 'M',
                512, 512, 'M'
            ],
            'vgg16': [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 256, 'M',
                512, 512, 512, 'M',
                512, 512, 512, 'M'
            ],
            'vgg19': [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 256, 256, 'M',
                512, 512, 512, 512, 'M',
                512, 512, 512, 512, 'M'
            ],
        }
        cfg = {'vgg11': 'A', 'vgg13': 'B', 'vgg16': 'D', 'vgg19': 'E'}[arch]
        weight_str = (
            f"models.{arch.upper()}_BN_Weights.DEFAULT" if batch_norm else
            f"models.{arch.upper()}_Weights.DEFAULT"
        )

        super().__init__(make_layers(self.convlayer_configs[arch], batch_norm))

        if pretrained:
            exec(
                f"self.load_state_dict("
                f"models.vgg._vgg("
                f"weights={weight_str},"
                f"batch_norm={batch_norm}, progress=False, cfg='D').state_dict())"
            )
            # print('Pretrained')

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if rm_fc:
            # Delete redundant fully-connected layer params, can save memory
            del self.classifier

        self.channel_progression = [
            self.convlayer_configs[arch][idx-1]
            for idx in [
                jdx for jdx, value in enumerate(self.convlayer_configs[arch]) if value=='M'
            ]
        ]

        # replace_indices = [4, 9, 16, 23, 30]
        # replace_modules = [Downsample(channel) for channel in self.channel_progression]
        # for idx, mod in zip(replace_indices, replace_modules):
            # exec(f"self.features[{idx}] = mod")
        insert_indices = [5, 12, 22, 32, 42] if batch_norm else [3, 8, 15, 22, 29]
        insert_modules = [
            Residual(PreNorm(channels, LinearAttention(idx, channels))).to(DEVICE)
            for idx, channels in enumerate(self.channel_progression)
        ]
        insert_modules[-1] = Residual(PreNorm(512, Attention(512))).to(DEVICE)
        insert_modules[-2] = Residual(PreNorm(512, Attention(512))).to(DEVICE)
        insert_modules[-3] = Residual(PreNorm(256, Attention(256))).to(DEVICE)
        #insert_modules[-4] = Residual(PreNorm(128, Attention(128))).to(DEVICE)
        for idx, mod in zip(insert_indices, insert_modules):
            exec(f"self.features[idx] = nn.Sequential(self.features[idx], mod)")

    def forward(self, x):
        """Forward pass"""
        output = {}
        # Get the output of each downsampling layer
        for idx, __ in enumerate(self.convlayer_ranges):
            for layer in range(self.convlayer_ranges[idx][0], self.convlayer_ranges[idx][1]):
                x = self.features[layer](x)
            output[f"x{idx+1}"] = x

        return output

def make_layers(configs, batch_norm=False):
    """Construct layers for vgg net using configs"""
    layers = []
    in_channels = 3
    for config in configs:
        if config == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, config, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(config), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = config
    return nn.Sequential(*layers)
