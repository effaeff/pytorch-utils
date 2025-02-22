"""Collection of custom layers"""

import math
import numpy as np
import einops
from functools import partial
from einops.layers.torch import Rearrange
from inspect import isfunction

from torchvision.transforms import GaussianBlur

from scipy.signal.windows import gaussian

import matplotlib.pyplot as plt

import cv2

from pytorchutils.globals import torch, nn, DEVICE
from torch.nn import functional as F

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

class CausalAttention(nn.Module):
    """
    Causal attention module
    """
    def __init__(self, dim, heads=8, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(self.hidden_dim, dim, 1)

        self.attn = nn.MultiheadAttention(self.hidden_dim, heads, batch_first=True)

    def forward(self, x):
        """Forward pass"""
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b h x y -> b (x y) h", h=self.hidden_dim), qkv
            # lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        # q = q * self.scale

        # sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)

        # mask = torch.tril(torch.ones(sim.size())).to(DEVICE)
        # sim = sim.masked_fill(mask == 0, float('-inf'))

        # attn = sim.softmax(dim=-1)

        # out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)

        # out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # mask = torch.ones(h*w, h*w).tril(diagonal=0)
        # mask = mask.masked_fill(mask == 0, -float('inf'))

        # out = self.attn(q, k, v, need_weights=False, attn_mask=mask, is_causal=True)[0]
        out = self.attn(q, k, v, need_weights=False)[0]


        # out = einops.rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        out = einops.rearrange(out, "b (x y) h -> b h x y", x=h, y=w, h=self.hidden_dim)
        return self.to_out(out)


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

class LinearAttention3d(nn.Module):
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
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        """Forward pass"""
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=d, y=h, z=w)

        out = self.to_out(out)

        return out#, context

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
    """Upsampling layer"""
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

class PreNorm3d(nn.Module):
    """
    Apply norm before func
        - References: https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    def __init__(self, dim, func):
        super().__init__()
        self.func = func
        # self.norm = nn.GroupNorm(8, dim)
        self.norm = LayerNorm3d(dim)

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
        """Forard pass"""
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class LayerNorm3d(nn.Module):
    """
    Layer norm module
        - References: https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        """Forard pass"""
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class CannyFilter(nn.Module):
    """Canny filter with learnable thresholds"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.threshold_high = nn.Parameter(torch.rand(1))
        self.threshold_low = nn.Parameter(torch.rand(1))
        # self.threshold_high = nn.Parameter(torch.tensor(0.98))

        self.strong = 1
        self.weak = 0

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(
            dim,
            dim,
            kernel_size=(1, filter_size),
            padding=(0, filter_size//2),
            groups=dim,
            bias=False
        )
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))

        self.gaussian_filter_vertical = nn.Conv2d(
            dim,
            dim,
            kernel_size=(filter_size, 1),
            padding=(filter_size//2, 0),
            groups=dim,
            bias=False
        )
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(
            dim,
            dim,
            kernel_size=sobel_filter.shape,
            padding=sobel_filter.shape[0]//2,
            groups=dim,
            bias=False
        )
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))

        self.sobel_filter_vertical = nn.Conv2d(
            dim,
            dim,
            kernel_size=sobel_filter.shape,
            padding=sobel_filter.shape[0]//2,
            groups=dim,
            bias=False
        )
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))

        filter_0 = np.array([
            [0, 0,  0],
            [0, 1, -1],
            [0, 0,  0]
        ])
        filter_45 = np.array([
            [0, 0,  0],
            [0, 1,  0],
            [0, 0, -1]
        ])
        filter_90 = np.array([
            [ 0, 0, 0],
            [ 0, 1, 0],
            [ 0,-1, 0]
        ])
        filter_135 = np.array([
            [ 0, 0, 0],
            [ 0, 1, 0],
            [-1, 0, 0]
        ])
        filter_180 = np.array([
            [ 0, 0, 0],
            [-1, 1, 0],
            [ 0, 0, 0]
        ])
        filter_225 = np.array([
            [-1, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 0]
        ])

        filter_270 = np.array([
            [0,-1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        filter_315 = np.array([
            [0, 0, -1],
            [0, 1,  0],
            [0, 0,  0]
        ])
        all_filters = np.stack(
            [
                filter_0,
                filter_45,
                filter_90,
                filter_135,
                filter_180,
                filter_225,
                filter_270,
                filter_315
            ]
        )

        self.directional_filter = nn.Conv2d(
            dim,
            8 * dim,
            kernel_size=filter_0.shape,
            padding=filter_0.shape[-1]//2,
            groups=dim,
            bias=False
        )
        self.directional_filter.weight.data.copy_(
            torch.from_numpy(
                np.stack(
                    [all_filters for __ in range(dim)]
                ).reshape((dim*8, *filter_0.shape))[:, None, ...]
            )
        )
        self.directional_filter.weight.requires_grad_(False)

        # self.edges_fc = nn.Conv2d(self.dim, 1, 1)

    def forward(self, img):
        """Forward pass"""
        blurred = self.gaussian_filter_horizontal(img)
        blurred = self.gaussian_filter_vertical(blurred)

        grad_x = self.sobel_filter_horizontal(blurred)
        grad_y = self.sobel_filter_vertical(blurred)

        # Thick edges
        grad_mag = torch.hypot(grad_x, grad_y)

        grad_orientation = torch.atan2(grad_y, grad_x) * 180 / torch.pi
        grad_orientation += 180.0
        grad_orientation = torch.round(grad_orientation / 45.0 ) * 45.0

        # Thin edges (non-max suppression)
        directional = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        thin_edges = grad_mag.clone()
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (inidices_positive==pos_i)
            is_oriented_i = is_oriented_i + (inidices_negative==neg_i)
            pos_directional = directional[:, pos_i::8]
            neg_directional = directional[:, neg_i::8]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # Get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0

            # Apply non-maximum suppression
            to_remove = (is_max == 0) * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholded = self.edges_fc(thin_edges)

        # Threshold
        # thresholded = thin_edges.clone()
        # thresholded = self.thresholding(thresholded)

        # return thresholded
        # return grad_mag
        return thin_edges

    def thresholding(self, img):
        """Double thresholding"""
        high = img.max() * self.threshold_high
        # low = high * self.threshold_low

        img[torch.where(img >= high)] = self.strong
        img[torch.where(img < high)] = 0

        # img[torch.where(img < low)] = 0.0
        # img[torch.where((img >= low) & (img < high))] = self.weak

        return img

    def hysteresis(self, img):
        """Check whether weak pixels can be considered strong"""
        out = img.clone()
        for b_idx, __ in enumerate(img):

            batch = img[b_idx].squeeze()

            for hdx in range(img.size()[-2]):
                for wdx in range(img.size()[-1]):
                    if batch[hdx, wdx] == self.weak:
                        try:
                            if self.strong in (
                                batch[hdx+1, wdx-1],
                                batch[hdx+1, wdx],
                                batch[hdx+1, wdx+1],
                                batch[hdx, wdx-1],
                                batch[hdx, wdx+1],
                                batch[hdx-1, wdx-1],
                                batch[hdx-1, wdx],
                                batch[hdx-1, wdx+1]
                            ):
                                out[b_idx, :, hdx, wdx] = self.strong
                            else:
                                out[b_idx, :, hdx, wdx] = 0
                        except IndexError as e:
                            continue
        return out

    def get_rotation_matrix_2d(self, center, angle, scale):
        """Get tensor of 2d rotation matrix"""
        alpha = scale * np.cos(angle)
        beta = scale * np.sin(angle)
        return torch.tensor([
            [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
            [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
        ])
