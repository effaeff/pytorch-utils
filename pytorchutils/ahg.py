"""Models for semantic segmentation based on VGG architectures"""

import copy

from pytorchutils.avgg import AVGGModel
from pytorchutils.vgg import VGGModel
from pytorchutils.layers import Attention, LinearAttention, PreNorm, Residual, CannyFilter
from pytorchutils.globals import nn, DEVICE, torch

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
        self.channel_to_vgg = nn.Conv2d(self.n_channels, 3, kernel_size=1)
        # self.channel_to_vgg = nn.Conv2d(self.n_channels, 3, kernel_size=3, padding=1)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.pretrained_net = AVGGModel(
            self.config
        )
        # self.pretrained_net = VGGModel(
            # self.config
        # )
        self.activation = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)

        # self.decode1 = nn.Sequential(
            # nn.ConvTranspose2d(512, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(512, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(512, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # Residual(PreNorm(512, LinearAttention(0, 512))),
            # # nn.Upsample(scale_factor=2, mode='nearest')
            # Upsample(512)
        # )
        # self.decode2 = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(256, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(256, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # Residual(PreNorm(256, LinearAttention(0, 256))),
            # # nn.Upsample(scale_factor=2, mode='nearest')
            # Upsample(256)
        # )
        # self.decode3 = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(128, 128, 3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # Residual(PreNorm(128, LinearAttention(0, 128))),
            # # nn.Upsample(scale_factor=2, mode='nearest')
            # Upsample(128)
        # )
        # self.decode4 = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(64, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # Residual(PreNorm(64, LinearAttention(0, 64))),
            # # nn.Upsample(scale_factor=2, mode='nearest')
            # Upsample(64)
        # )
        # self.decode5 = nn.Sequential(
            # nn.ConvTranspose2d(64, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(32, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # Residual(PreNorm(32, LinearAttention(0, 32))),
            # # nn.Upsample(scale_factor=2, mode='nearest')
            # Upsample(32)
        # )

        #self.attn1 = Residual(PreNorm(512, LinearAttention(0, 512)))
        #self.attn2 = Residual(PreNorm(256, LinearAttention(0, 256)))
        #self.attn3 = Residual(PreNorm(128, LinearAttention(0, 128)))
        self.attn4 = Residual(PreNorm(64, LinearAttention(0, 64)))
        self.attn1 = Residual(PreNorm(512, Attention(512)))
        self.attn2 = Residual(PreNorm(256, Attention(256)))
        self.attn3 = Residual(PreNorm(128, Attention(128)))
        #self.attn4 = Residual(PreNorm(64, Attention(64)))
        self.attn5 = Residual(PreNorm(32, LinearAttention(0, 32)))

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)

        self.canny = CannyFilter(self.output_size)

        # self.skip_conv1 = nn.Conv2d(1024, 512, 1)
        # self.skip_conv2 = nn.Conv2d(512, 256, 1)
        # self.skip_conv3 = nn.Conv2d(256, 128, 1)
        # self.skip_conv4 = nn.Conv2d(128, 64, 1)

        self.classifier = nn.Conv2d(32, self.output_size, kernel_size=1)

        self.edges_fc = nn.Conv2d(self.output_size, 1, 1)

    def forward(self, inp):
        """Forward pass"""
        if self.n_channels != 3:
            inp = torch.sigmoid(self.channel_to_vgg(inp))
        inp = self.norm(inp)
        b, c, h, w = inp.size()
        vgg_out = self.pretrained_net(inp)
        x_5 = vgg_out['x5']  # size = (N, 512, H/32, W/32)
        x_4 = vgg_out['x4']  # size = (N, 512, H/16, W/16)
        x_3 = vgg_out['x3']  # size = (N, 256, H/8,  W/8)
        x_2 = vgg_out['x2']  # size = (N, 128, H/4,  W/4)
        x_1 = vgg_out['x1']  # size = (N, 64, H/2,  W/2)

        # Size = (N, 512, H/16, W/16)
        pred_out = self.activation(self.deconv1(x_5))
        pred_out = self.attn1(pred_out)
        # Element-wise add, size = (N, 512, H/16, W/16)
        pred_out = self.bn1(pred_out + x_4)
        # pred_out = self.bn1(self.skip_conv1(torch.cat((pred_out, x_4), dim=1)))
        pred_out = self.activation(pred_out)
        # Size = (N, 256, H/8, W/8)
        pred_out = self.activation(self.deconv2(pred_out))
        pred_out = self.attn2(pred_out)
        # Element-wise add, size = (N, 256, H/8, W/8)
        pred_out = self.bn2(pred_out + x_3)
        # pred_out = self.bn2(self.skip_conv2(torch.cat((pred_out, x_3), dim=1)))
        pred_out = self.activation(pred_out)
        # Size = (N, 128, H/4, W/4)
        pred_out = self.activation(self.deconv3(pred_out))
        pred_out = self.attn3(pred_out)
        # Element-wise add, size = (N, 128, H/4, W/4)
        pred_out = self.bn3(pred_out + x_2)
        # pred_out = self.bn3(self.skip_conv3(torch.cat((pred_out, x_2), dim=1)))
        pred_out = self.activation(pred_out)
        # Size = (N, 64, H/2, W/2)
        pred_out = self.activation(self.deconv4(pred_out))
        pred_out = self.attn4(pred_out)
        # Element-wise add, size = (N, 64, H/2, W/2)
        pred_out = self.bn4(pred_out + x_1)
        # pred_out = self.bn4(self.skip_conv4(torch.cat((pred_out, x_1), dim=1)))
        pred_out = self.activation(pred_out)
        # Size = (N, 32, H, W)
        pred_out = self.activation(self.deconv5(pred_out))
        pred_out = self.attn5(pred_out)

        # Size = (N, output_size, H/1, W/1)
        pred_out = self.classifier(pred_out)

        edges = self.canny(pred_out)
        edges = self.edges_fc(edges).view(b, h, w)

        return pred_out, edges

    # def forward(self, inp):
        # if self.n_channels != 3:
            # inp = torch.sigmoid(self.channel_to_vgg(inp))
        # inp = self.norm(inp)
        # vgg_out = self.pretrained_net(inp)
        # x_5 = vgg_out['x5']  # size = (N, 512, H/32, W/32)
        # x_4 = vgg_out['x4']  # size = (N, 512, H/16, W/16)
        # x_3 = vgg_out['x3']  # size = (N, 256, H/8,  W/8)
        # x_2 = vgg_out['x2']  # size = (N, 128, H/4,  W/4)
        # x_1 = vgg_out['x1']  # size = (N, 64, H/2,  W/2)
        # pred_out = self.bn1(self.decode1(x_5) + x_4)
        # pred_out = self.bn2(self.decode2(pred_out) + x_3)
        # pred_out = self.bn3(self.decode3(pred_out) + x_2)
        # pred_out = self.bn4(self.decode4(pred_out) + x_1)
        # pred_out = self.decode5(pred_out)
        # pred_out = self.classifier(pred_out)
        # return pred_out

    def inp2rgb(self, inp):
        """Evaluate self.channel_to_vgg to get rgb channels for inp"""
        return torch.sigmoid(self.channel_to_vgg(inp))
