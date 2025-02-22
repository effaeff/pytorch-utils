"""Models for VGG11/13/16/19 architectures for the usage as backbone for FCN models"""

import copy
from torchvision import models
from pytorchutils.globals import nn

class VGGModel(models.vgg.VGG):
    """
    VGG backbone cropped before fully connected layers.
    References:
        - https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    """
    def __init__(self, config, requires_grad=True, rm_fc=True):
        self.config = copy.deepcopy(config)
        arch = self.config.get('arch', 'vgg16')
        pretrained = self.config.get('pretrained', True)

        self.convlayer_ranges = {
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
        super().__init__(make_layers(self.convlayer_configs[arch]))

        if pretrained:
            exec(
                f"self.load_state_dict("
                f"models.{arch}(weights=models.{arch.upper()}_Weights.DEFAULT).state_dict())"
            )
            print('Pretrained')

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if rm_fc:
            # Delete redundant fully-connected layer params, can save memory
            del self.classifier

    def forward(self, x):
        """Forward pass"""
        output = {}
        # Get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx, __ in enumerate(self.convlayer_ranges):
            for layer in range(self.convlayer_ranges[idx][0], self.convlayer_ranges[idx][1]):
                # print(f"{idx}\t{layer}")
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
