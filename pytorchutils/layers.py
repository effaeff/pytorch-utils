"""Collection of custom layers"""

import math
import numpy as np
import einops

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

class SelfAttention(nn.Module):
    """
    Self attention layer
    Referrences:
        - https://arxiv.org/abs/1805.08318
    """
    def __init__(self, n_channels, image_size, n_patches, embedding_size, patchify=False):
        super().__init__()
        self.patchify = patchify
        # Use 1D convolutions to process data of arbitrary spacial dimensions
        self.query, self.key, self.value = [
            nn.Conv1d(n_channels, c, kernel_size=1)
            for c in [
                n_channels//8 if n_channels//8 > 0 else n_channels,
                n_channels//8 if n_channels//8 > 0 else n_channels,
                n_channels
            ]
        ]
        self.gamma = nn.Parameter(torch.zeros(1))

        self.sequentializer = ImageSequentializer(n_patches, image_size, embedding_size)

        self.upscale = nn.Linear(embedding_size, int(image_size[0] * image_size[1]))

    def forward(self, inp):
        size = inp.size()
        if self.patchify:
            inp = self.sequentializer(inp)
        else:
            # Reshape spacial dimensions to 1
            inp = inp.view(*size[:2], -1) # (B, C, N)

        query, key, value = self.query(inp), self.key(inp), self.value(inp)

        energy = torch.bmm(query.transpose(1, 2), key)
        attention = nn.functional.softmax(energy, dim=1) # (B, N, N)

        out = self.gamma * torch.bmm(value, attention) # + inp

        if self.patchify:
            out = self.upscale(out)

        return torch.cat((inp, out), dim=1)
        # return out.view(*size)

class ImageSequentializer(nn.Module):

    def __init__(self, num_patches, image_size, embedding_size):
        super(ImageSequentializer, self).__init__()
        self.num_patches = num_patches
        self.linear = nn.Linear(
            int(image_size[0] // num_patches[0] *
            image_size[1] // num_patches[0] *
            num_patches[0] * num_patches[1]),
            embedding_size
        )
        self.pos_embedding = PositionalEncoding(embedding_size)

    def forward(self, x):

        patches = einops.rearrange(
            x,
            "b c (p1 h) (p2 w) -> b c (p1 p2) h w",
            p1=self.num_patches[0],
            p2=self.num_patches[1]
        )

        # DEBUG
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(16, 16)
        # for i in range(self.num_patches[0]):
            # for j in range(self.num_patches[1]):

                # ax[i, j].imshow(
                    # patches[0, 0, i * self.num_patches[0] + j].cpu().detach().numpy(),
                    # vmin=0,
                    # vmax=1
                # )
        # plt.show()

        flat_patches = einops.rearrange(
            patches,
            "b c p h w -> b c (p h w)"
        )

        embeddings = self.linear(flat_patches)
        embeddings = self.pos_embedding(embeddings)

        return embeddings


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
