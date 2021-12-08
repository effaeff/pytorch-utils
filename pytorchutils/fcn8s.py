"""Models for semantic segmentation based on VGG architectures"""

import copy

from pytorchutils.vgg import VGGModel
from pytorchutils.globals import nn


class FCNModel(nn.Module):
    """Actual class for FCN8s model using the VGG architecture as backbone"""
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)

        self.output_size = self.config.get('output_size', 0)
        if self.output_size == 0:
            raise ValueError("Error: No output size defined.")

        self.pretrained_net = VGGModel(
            self.config['models_dir'],
            self.config.get('arch', 'vgg16'),
            self.config.get('pretrained', True)
        )
        self.activation = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.output_size, kernel_size=1)

        ###################test##############################
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        ########fcn part
        self.score_fc = nn.Conv2d(4096, 512, 1)


        # self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        # self.drop6 = nn.Dropout2d()
        # self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        # self.drop7 = nn.Dropout2d()

        # self.score_fn = nn.Conv2d(4096, self.output_size, kernel_size=1)
        # self.score_pool4 = nn.Conv2d(512, self.output_size, 1)
        # self.score_pool3 = nn.Conv2d(256, self.output_size, 1)

        # self.upscore = nn.ConvTranspose2d(self.output_size, self.output_size, 9, stride=8, padding=1, output_padding=1)
        # self.upscore4 = nn.ConvTranspose2d(self.output_size, self.output_size, 3, stride=2, padding=1, output_padding=1)
        # self.upscore5 = nn.ConvTranspose2d(self.output_size, self.output_size, 3, stride=2, padding=1, output_padding=1)

    # def forward(self, inp):
        # vgg_out = self.pretrained_net(inp)
        # x_5 = vgg_out['x5']  # size = (N, 512, H/32, W/32)
        # x_4 = vgg_out['x4']  # size = (N, 512, H/16, W/16)
        # x_3 = vgg_out['x3']  # size = (N, 256, H/8,  W/8)

        # pred_out = self.activation(self.fc6(x_5))
        # pred_out = self.drop6(pred_out)

        # pred_out = self.activation(self.fc7(pred_out))
        # pred_out = self.drop7(pred_out)

        # pred_out = self.score_fn(pred_out)

        # upscore5 = self.upscore5(pred_out)

        # score4 = self.score_pool4(x_4)
        # score4 += upscore5

        # upscore4 = self.upscore4(score4)

        # score3 = self.score_pool3(x_3)
        # score3 += upscore4

        # out = self.upscore(score3)
        # return out

    def forward(self, inp):
        """Forward pass through vgg and upscaling layers"""
        vgg_out = self.pretrained_net(inp)
        x_5 = vgg_out['x5']  # size = (N, 512, H/32, W/32)
        x_4 = vgg_out['x4']  # size = (N, 512, H/16, W/16)
        x_3 = vgg_out['x3']  # size = (N, 256, H/8,  W/8)

        # Size = (N, 512, H/16, W/16)
        pred_out = self.activation(self.deconv1(x_5))
        # Element-wise add, size = (N, 512, H/16, W/16)
        pred_out = self.bn1(pred_out + x_4)
        # Size = (N, 256, H/8, W/8)
        pred_out = self.activation(self.deconv2(pred_out))
        # Element-wise add, size = (N, 256, H/8, W/8)
        pred_out = self.bn2(pred_out + x_3)
        # Size = (N, 128, H/4, W/4)
        pred_out = self.bn3(self.activation(self.deconv3(pred_out)))
        # Size = (N, 64, H/2, W/2)
        pred_out = self.bn4(self.activation(self.deconv4(pred_out)))
        # Size = (N, 32, H, W)
        pred_out = self.bn5(self.activation(self.deconv5(pred_out)))
        # Size = (N, output_size, H/1, W/1)
        pred_out = self.classifier(pred_out)

        # Size = (N, output_size, H/1, W/1)
        return pred_out
