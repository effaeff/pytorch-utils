"""Models for semantic segmentation based on VGG architectures"""

import copy

from vgg import VGGModel
from globals import nn

class FCNModel(nn.Module):
    """Actual class for FCN8s model using the VGG architecture as backbone"""
    def __init__(self, config):
        super(FCNModel, self).__init__()
        self.config = copy.deepcopy(config)

        self.output_size = self.config.get('output_size', 0)
        if self.output_size == 0:
            raise ValueError("Error: No output size defined.")

        self.backbone = VGGModel(
            self.config.get('arch', 'vgg16'),
            self.config.get('pretrained', True)
        )
        self.activation = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.output_size, kernel_size=1)

    def forward(self, inp):
        """Forward pass through vgg and upscaling layers"""
        vgg_out = self.backbone(inp)
        x5 = vgg_out['x5']  # size = (N, 512, H/32, W/32)
        x4 = vgg_out['x4']  # size = (N, 512, H/16, W/16)
        x3 = vgg_out['x3']  # size = (N, 256, H/8,  W/8)

        # Size = (N, 512, H/16, W/16)
        pred_out = self.relu(self.deconv1(x5))
        # Element-wise add, size = (N, 512, H/16, W/16)
        pred_out = self.bn1(pred_out + x4)
        # Size = (N, 256, H/8, W/8)
        pred_out = self.relu(self.deconv2(pred_out))
        # Element-wise add, size = (N, 256, H/8, W/8)
        pred_out = self.bn2(pred_out + x3)
        # Size = (N, 128, H/4, W/4)
        pred_out = self.bn3(self.relu(self.deconv3(pred_out)))
        # Size = (N, 64, H/2, W/2)
        pred_out = self.bn4(self.relu(self.deconv4(pred_out)))
        # Size = (N, 32, H, W)
        pred_out = self.bn5(self.relu(self.deconv5(pred_out)))
        # Size = (N, output_size, H/1, W/1)
        pred_out = self.classifier(pred_out)

        # Size = (N, output_size, H/1, W/1)
        return pred_out
