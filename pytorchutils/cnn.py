"""Convolutional neural network model"""

from functools import reduce
import numpy as np

import misc
from pytorchutils.basic_model import BasicModel
from pytorchutils.globals import nn, torch

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

        self.dim = self.config.get('dimension', 2)

        # Calculate dimension of last conv output,
        # assuming conv layers do not change input dimension and pool reduces it by half

        last_dim = self.channels[-1]
        for d in range(self.dim):
            last_dim *= self.input_size[d] // (2**self.nb_layers)

        self.fc_units = []
        self.fc_units.append(last_dim)

        if isinstance(self.nb_units, int) and self.nb_units is not None:
            self.fc_units.append(self.nb_units)
        elif isinstance(self.nb_units, (list, np.ndarray)):
            for unit_value in self.nb_units:
                self.fc_units.append(unit_value)
        self.fc_units.append(self.output_size)

        self.dropout_conv = nn.Dropout(p=self.config.get('dropout_rate_conv', 0.0))

        self.conv_layer
        self.fc_layer

    @misc.lazy_property
    def conv_layer(self):
        """Property for a set of conv->activation->maxpool layers"""
        conv_layer = nn.ModuleList()
        for layer_idx in range(self.nb_layers):
            layer = nn.Sequential(
                getattr(nn, f'Conv{self.dim}d')(
                    self.channels[layer_idx],
                    self.channels[layer_idx + 1],
                    kernel_size=self.config.get('kernel_size_conv', 3),
                    stride=self.config.get('stride_conv', 1),
                    padding=self.config.get('padding_conv', 1),
                    dilation=self.config.get('dilation_conv', 1)
                ),
                getattr(nn, f'BatchNorm{self.dim}d')(self.channels[layer_idx + 1]),
                self.activation,
                self.dropout_conv,
                getattr(nn, f'MaxPool{self.dim}d')(
                    kernel_size=self.config.get('kernel_size_pool', 2),
                    padding=self.config.get('padding_pool', 0),
                    stride=self.config.get('stride_pool', 2),
                    dilation=self.config.get('dilation_pool', 1)
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
        pred_out = torch.flatten(pred_out, start_dim=1)

        pred_out = self.dropout(pred_out)
        for layer in self.fc_layer[:-1]:
            pred_out = layer(pred_out)
            pred_out = self.activation(pred_out)
            pred_out = self.dropout(pred_out)

        pred_out = self.fc_layer[-1](pred_out)
        return pred_out
