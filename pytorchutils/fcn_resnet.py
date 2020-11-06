"""Models for semantic segmentation based on ResNet architectures"""

import copy
from torchvision import models

from globals import nn

class FCNModel(nn.Module):
    """Actual class for FCN model using the ResNet architecture as backbone"""
    def __init__(self, config):
        super(FCNModel, self).__init__()
        self.config = copy.deepcopy(config)

        self.output_size = self.config.get('output_size', 0)
        if self.output_size == 0:
            raise ValueError("Error: No output size defined.")

        self.model = getattr(models, self.config.get('arch', 'fcn_resnet50'))(
            self.config.get('pretrained', True),
            self.output_size)
        )

    def forward(self, inp):
        """Forward pass through FCN"""
        return self.model(inp)['out']
