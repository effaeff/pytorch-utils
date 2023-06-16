"""Models for semantic segmentation based on ResNet architectures"""

import re
import copy
from torchvision import models

from pytorchutils.globals import nn

class Segmentation(nn.Module):
    """Actual class for FCN model using the ResNet architecture as backbone"""
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.arch = self.config.get('arch', 'fcn_resnet50')
        self.n_channels = self.config.get('n_channels', 3)

        self.output_size = self.config.get('output_size', 0)
        if self.output_size == 0:
            raise ValueError("Error: No output size defined.")

        self.channel_to_resnet = nn.Conv2d(self.n_channels, 3, 1)

        self.model = getattr(models.segmentation, self.arch)(
            weights='DEFAULT'#,
            # num_classes=self.output_size
        )
        if self.arch == 'deeplabv3_resnet50':
            self.model.classifier[4] = nn.Conv2d(256, self.output_size, 1)
        elif self.arch == 'lraspp_mobilenet_v3_large':
            self.model.classifier.low_classifier = nn.Conv2d(40, self.output_size, 1)
            self.model.classifier.high_classifier = nn.Conv2d(128, self.output_size, 1)
        elif self.arch == 'fcn_resnet50':
            self.model.classifier[4] = nn.Conv2d(512, self.output_size, 1)

    def forward(self, inp):
        """Forward pass through FCN"""
        inp = self.channel_to_resnet(inp)
        return self.model(inp)['out']
