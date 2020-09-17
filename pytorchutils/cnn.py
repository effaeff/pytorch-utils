"""Convolutional neural network model"""

import numpy as np
from functools import reduce

import misc
from basic_model import BasicModel
from globals import nn

class CNNModel(BasicModel):
    """
    Actual class for CNN model.
    """
    def __init__(self, config):
        BasicModel.__init__(self, config)
        self.channels = self.config.get('channels', None)
        if self.channels is None:
            raise ValueError(
                "Error: No channels for convolutions defined. "
                "These values have to be specified for all CNN architectures."
            )
        if len(self.channels) != self.nb_layers + 1:
            raise ValueError(
                "Error: Invalid number of specified convolution channels. "
                "The length of channels should equal nb_layers + 1."
            )

        self.image_size = self.config.get('image_size', None)
        if self.image_size is None:
            raise ValueError(
                "Error: No image_size specified. "
                "This is mandatory for using CNN architectures."
            )
        if np.shape(self.image_size) != (2,):
            raise ValueError(
                "Error: Invalid shape of image_size. "
                "The provided shape should be (2,)."
            )

        self.fc_units = []
        self.fc_units.append(self.image_size[0] * self.image_size[1] * self.channels[-1])
        if isinstance(self.nb_units, int) and self.nb_units != 0:
            self.fc_units.append(self.nb_units)
        elif isinstance(self.nb_units, (list, np.ndarray)):
            for unit_value in self.nb_units:
                self.fc_units.append(unit_value)
        self.fc_units.append(self.output_size)

        self.conv_layer
        self.fc_layer

    @misc.lazy_property
    def conv_layer(self):
        """Property for a set of conv->activation->maxpool layers"""
        conv_layer = nn.ModuleList()
        for layer_idx in range(self.nb_layers):
            layer = nn.Sequential(
                nn.Conv2d(
                    self.channels[layer_idx],
                    self.channels[layer_idx + 1],
                    kernel_size=self.config.get('kernel_size_conv', 3),
                    stride=self.config.get('stride_conv', 1),
                    padding=self.config.get('padding', 1)
                ),
                self.activation,
                nn.MaxPool2d(
                    kernel_size=self.config.get('kernel_size_pool', 2),
                    stride=self.config.get('stride_pool', 2)
                )
            )
            conv_layer.append(layer)
        return conv_layer

    @misc.lazy_property
    def fc_layer(self):
        """Property for a set of fully connected layer"""
        fc_layer = nn.ModuleList()
        for unit_idx, __ in enumerate(self.fc_units[:-1]):
            fc_layer.append(nn.Linear(self.fc_units[unit_idx], self.fc_units[unit_idx + 1]))
        return fc_layer

    def forward(self, inp):
        """Forward pass through convolution and fully connected layer"""
        pred_out = reduce(lambda x, y: y(x), self.conv_layer, inp)
        for layer in self.fc_layer[:-1]:
            pred_out = layer(pred_out)
            pred_out = self.dropout(pred_out)
        pred_out = self.fc_layer[-1](pred_out)
        return pred_out
