"""Models for semantic segmentation based on VGG architectures"""

import copy

from pytorchutils.avgg import AVGGModel
from pytorchutils.layers import Attention, LinearAttention, PreNorm, Residual
from pytorchutils.globals import nn, DEVICE

from torchvision import transforms

class AHGModel(nn.Module):
    """Hourglass model using attention blocks in encoder for semantic segmentation"""
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)

        self.n_channels = self.config.get('n_channels', 3)
        self.output_size = self.config.get('output_size', 0)
        if self.output_size == 0:
            raise ValueError("Error: No output size defined.")

        # Transform input channels to 3, so that VGG weights can be used
        self.channel_to_vgg = nn.Conv2d(self.n_channels, 3, kernel_size=3, padding=1)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.pretrained_net = AVGGModel(
            self.config
        )
        self.activation = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)
        # self.gn1 = nn.GroupNorm(1, 512)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        # self.gn2 = nn.GroupNorm(1, 256)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        # self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.output_size, kernel_size=1)

        self.self_attn = [
            Residual(PreNorm(channels, LinearAttention(idx, channels))).to(DEVICE)
            for idx, channels in enumerate([512, 256, 128, 64, 32])
        ]
        # self.self_attn[0] = Residual(PreNorm(512, Attention(512))).to(DEVICE)
        # self.self_attn[1] = Residual(PreNorm(256, Attention(256))).to(DEVICE)

    def forward(self, inp):
        """Forward pass through vgg and upscaling layers"""
        if self.n_channels != 3:
            inp = self.activation(self.channel_to_vgg(inp))
        inp = self.norm(inp)
        vgg_out = self.pretrained_net(inp)
        # vgg_out, attn = self.pretrained_net(inp)
        x_5 = vgg_out['x5']  # size = (N, 512, H/32, W/32)
        x_4 = vgg_out['x4']  # size = (N, 512, H/16, W/16)
        x_3 = vgg_out['x3']  # size = (N, 256, H/8,  W/8)

        # Size = (N, 512, H/16, W/16)
        pred_out = self.activation(self.deconv1(x_5))
        pred_out = self.self_attn[0](pred_out)
        # Element-wise add, size = (N, 512, H/16, W/16)
        pred_out = self.bn1(pred_out + x_4)
        # Size = (N, 256, H/8, W/8)
        pred_out = self.activation(self.deconv2(pred_out))
        pred_out = self.self_attn[1](pred_out)
        # Element-wise add, size = (N, 256, H/8, W/8)
        pred_out = self.bn2(pred_out + x_3)
        # Size = (N, 128, H/4, W/4)
        pred_out = self.activation(self.deconv3(pred_out))
        pred_out = self.self_attn[2](pred_out)
        # Size = (N, 64, H/2, W/2)
        pred_out = self.activation(self.deconv4(pred_out))
        pred_out = self.self_attn[3](pred_out)
        # Size = (N, 32, H, W)
        pred_out = self.activation(self.deconv5(pred_out))
        pred_out = self.self_attn[4](pred_out)
        # Size = (N, output_size, H/1, W/1)
        pred_out = self.classifier(pred_out)

        # Size = (N, output_size, H/1, W/1)
        return pred_out#, attn
