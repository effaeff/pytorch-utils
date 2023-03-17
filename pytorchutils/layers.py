"""Collection of custom layers"""

import math
import numpy as np
import einops
from functools import partial
from einops.layers.torch import Rearrange
from inspect import isfunction

import matplotlib.pyplot as plt

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
        if alpha * (length-1) / 2 >= idx >= 0:
            return self.haversin(math.pi * ((2*idx) / (alpha*(length-1)) - 1))
        if (length-1) * (1-(alpha/2)) >= idx >= alpha * (length-1) / 2:
            return 1
        if (length-1) >= idx >= (length-1) * (1-(alpha/2)):
            return self.haversin(math.pi * ((2*idx) / (alpha*(length-1)) - (2/alpha) + 1))

    def welch(self, idx, length):
        """Welch window function"""
        return 1 - math.pow((idx - (length-1) / 2) / ((length-1) / 2), 2)

    def heaviside(self, value):
        """Heaviside function"""
        res = 0 if value < 0 else 1
        return res

class WSConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = einops.reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = einops.reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return nn.functional.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

class WSConvTranspose2d(nn.ConvTranspose2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = einops.reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = einops.reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return nn.functional.conv_transpose2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation
        )

class Attention(nn.Module):
    """
    Attention module
        - References: https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    def __init__(self, dim, heads=8, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """Forward pass"""
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = einops.rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    """
    Linear attention module
        - References: https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    def __init__(self, layer_idx, dim, heads=8, dim_head=32):
        super().__init__()
        self.layer_idx = layer_idx
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """Forward pass"""
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        # self.plot_attn(einops.rearrange(k, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)[0], "attn")
        # self.plot_attn(x[0], "inp")

        out = self.to_out(out)

        return out#, context

    def plot_attn(self, attn, f_name):
        """Method for plotting attention map"""
        fig, axs = plt.subplots(8, 8)
        for idx, channel in enumerate(attn[:64]):
            axs[idx//8][idx%8].imshow(channel.detach().cpu().numpy(), cmap='Greys')
        plt.setp(np.reshape(axs, (-1,)), xticks=[], yticks=[])
        plt.tight_layout(pad=0)
        plt.savefig(f"{f_name}_layer{self.layer_idx}.png", dpi=600)

class PatchEmbedding(nn.Module):
    """Sequentialize images through lineralized patches"""
    def __init__(self, num_patches, image_size, embedding_size):
        super().__init__()
        self.num_patches = num_patches
        self.linear = nn.Linear(
            int(image_size[0] // num_patches[0] *
            image_size[1] // num_patches[0] *
            num_patches[0] * num_patches[1]),
            embedding_size
        )

    def forward(self, inp):
        """Forwars pass"""
        patches = einops.rearrange(
            inp,
            "b c (p1 h) (p2 w) -> b c (p1 p2) h w",
            p1=self.num_patches[0],
            p2=self.num_patches[1]
        )

        flat_patches = einops.rearrange(
            patches,
            "b c p h w -> b c (p h w)"
        )

        embeddings = self.linear(flat_patches)

        return embeddings

class PositionalEncoding(nn.Module):
    """Positional encoding"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        p_e = torch.zeros(max_len, 1, d_model)
        p_e[:, 0, 0::2] = torch.sin(position * div_term)
        p_e[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', p_e)

    def forward(self, inp):
        """
        Forward pass
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        inp = inp + self.pe[:inp.size(0)]
        return self.dropout(inp)

def Upsample(dim):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim, 3, padding=1),
    )

class Residual(nn.Module):
    """
    Residual connection around func
        - References: https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, inp, *args, **kwargs):
        """Forward pass"""
        return self.func(inp, *args, **kwargs) + inp

class PreNorm(nn.Module):
    """
    Apply norm before func
        - References: https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    def __init__(self, dim, func):
        super().__init__()
        self.func = func
        # self.norm = nn.GroupNorm(8, dim)
        self.norm = LayerNorm(dim)

    def forward(self, inp):
        """Forward pass"""
        inp = self.norm(inp)
        return self.func(inp)

class LayerNorm(nn.Module):
    """
    Layer norm module
        - References: https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g
