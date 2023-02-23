"""Collection of custom layers"""

import math
import numpy as np

from pytorchutils.globals import torch, nn, DEVICE

class WindowingLayer(nn.Module):
    """Class for sparse model"""
    def __init__(self, input_size, nb_hidden, window='welch'):
        super().__init__()

        self.nb_hidden = nb_hidden
        self.window = window
        self.input_size = input_size
        self.encoder = nn.Linear(input_size, nb_hidden, bias=False)
        self.dist_weights = self.init_dist_matrix()

    def init_dist_matrix(self):
        """Initialize distance-based weight matrix"""
        weights = torch.zeros(self.nb_hidden, self.input_size).to(DEVICE)
        for idx in range(weights.size(0)):
            for jdx in range(weights.size(1)):
                weights[idx][jdx] = getattr(self, self.window)(
                    abs(idx - jdx),
                    weights.size(1) * 2
                )
        return weights

    def forward(self, inp):
        """Forward pass"""
        # inp is supposed to be ordered row-major,
        # i.e., dimension looks like: [batch, sample]
        w_abs = torch.abs(self.encoder.weight)
        d_loss = torch.sum(w_abs * self.dist_weights) * (1 / (self.input_size * self.nb_hidden))
        out = self.encoder(inp)

        return out, d_loss

    def blackman(self, idx, length):
        """Blackman filter function"""
        alpha = 0.16
        a_0 = (1 - alpha) / 2
        a_1 = 1 / 2
        a_2 = alpha / 2

        return (
            a_0
            - a_1 * math.cos((2*math.pi*idx) / (length-1))
            + a_2 * math.cos((4*math.pi*idx) / (length-1))
        )

    def haversin(self, phi):
        """Haversin function"""
        return (1-math.cos(phi)) / 2

    def lanczos(self, idx, length):
        """Lanczos filter function"""
        return np.sinc((2*math.pi*idx) / (length-1) - 1)

    def tukey(self, idx, length):
        """
        Tukey window function.
        For alpha = 0: rectangle
        For alpha = 1: hann
        """
        alpha = 0.5
        if idx >= 0 and idx <= alpha * (length-1) / 2:
            return self.haversin(math.pi * ((2*idx) / (alpha*(length-1)) - 1))
        elif idx >= alpha * (length-1) / 2 and idx <= (length-1) * (1-(alpha/2)):
            return 1
        elif idx >= (length-1) * (1-(alpha/2)) and idx <= (length-1):
            return self.haversin(math.pi * ((2*idx) / (alpha*(length-1)) - (2/alpha) + 1))

    def welch(self, idx, length):
        """Welch window function"""
        return 1 - math.pow((idx - (length-1) / 2) / ((length-1) / 2), 2)

    def heaviside(self, value):
        """Heaviside function"""
        res = 0 if value < 0 else 1
        return res
