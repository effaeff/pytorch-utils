"""Variational autoencoder model"""

from functools import reduce
import numpy as np

import misc
from pytorchutils.basic_model import BasicModel
from pytorchutils.globals import nn, torch, DEVICE

class VariationalEncoder(BasicModel):
    """Actual class for VAE model."""
    def __init__(self, config):
        BasicModel.__init__(self, config)

        # Convolutions are used for encoder layer by default
        self.channels = self.config.get('channels', None)
        print(self.channels)
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
        self.dim = self.config.get('dimension', 1)

        # self.units = [self.input_size]
        # if isinstance(self.nb_units, int) and self.nb_units is not None:
            # self.units.append(self.nb_units)
        # elif isinstance(self.nb_units, (list, np.ndarray)):
            # for value in self.nb_units:
                # self.units.append(value)

        self.normal_dist = torch.distributions.Normal(0, 1)
        self.normal_dist.loc = self.normal_dist.loc.to(DEVICE)
        self.normal_dist.scale = self.normal_dist.scale.to(DEVICE)

        self.kld = 0

        self.encoder_layer
        self.fc_mu
        self.fc_sigma

    # @misc.lazy_property
    # def encoder_layer(self):
        # """Property for hidden layer of encoder"""
        # encoder_layer = nn.ModuleList()
        # for unit_idx, __ in enumerate(self.units[:-1]):
            # encoder_layer.append(
                # nn.Sequential(
                    # nn.Linear(self.units[unit_idx], self.units[unit_idx + 1]),
                    # self.activation
                # )
            # )
        # return encoder_layer

    @misc.lazy_property
    def encoder_layer(self):
        """Property for a set of conv->activation->maxpool layers"""
        encoder_layer = nn.ModuleList()
        for layer_idx in range(self.nb_layers):
            layer = nn.Sequential(
                getattr(nn, f'Conv{self.dim}d')(
                    self.channels[layer_idx],
                    self.channels[layer_idx + 1],
                    kernel_size=self.config.get('kernel_size_conv', 3),
                    stride=self.config.get('stride_conv', 1),
                    padding=self.config.get('padding_conv', 1),
                    dilation=self.config.get('dilation_conv', 0)
                ),
                self.activation,
                getattr(nn, f'MaxPool{self.dim}d')(
                    kernel_size=self.config.get('kernel_size_pool', 2),
                    padding=self.config.get('padding_pool', 0),
                    stride=self.config.get('stride_pool', 2),
                    dilation=self.config.get('dilation_pool', 1)
                )
            )
            encoder_layer.append(layer)
        return encoder_layer

    @misc.lazy_property
    def fc_mu(self):
        """Property for layer to estimate mu of gaussian distribution"""
        # output_size defines latent dimension
        return nn.Linear(self.channels[-1], self.output_size)

    @misc.lazy_property
    def fc_sigma(self):
        """Property for layer to estimate sigma of gaussian distribution"""
        # output_size defines latent dimension
        return nn.Linear(self.channels[-1], self.output_size)

    def forward(self, inp):
        """Forward pass"""
        inp = reduce(lambda x, y: y(x), self.encoder_layer, inp)
        inp = torch.squeeze(inp)
        mu = self.fc_mu(inp)
        sigma = torch.exp(self.fc_sigma(inp))

        latent = mu + sigma * self.normal_dist.sample(mu.shape) # Reparametrization trick
        self.kld = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return latent
