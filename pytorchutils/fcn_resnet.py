"""Models for semantic segmentation based on ResNet architectures"""

import copy
from torchvision import models

from pytorchutils.globals import nn

class FCNModel(nn.Module):
    """Actual class for FCN model using the ResNet architecture as backbone"""
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)

        self.output_size = self.config.get('output_size', 0)
        if self.output_size == 0:
            raise ValueError("Error: No output size defined.")

        self.model = getattr(models.segmentation, self.config.get('arch', 'fcn_resnet50'))(
            self.config.get('pretrained', True),
            num_classes=self.output_size
        )

    def forward(self, inp):
        """Forward pass through FCN"""
        return self.model(inp)['out']
