"""Collection of custom layers"""

import math
import numpy as np
import einops
from functools import partial
from einops.layers.torch import Rearrange
from inspect import isfunction

from torchvision.transforms import GaussianBlur

import matplotlib.pyplot as plt

import cv2

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

def get_rotation_matrix_2d(center, angle, scale):
    alpha = scale * np.cos(angle)
    beta = scale * np.sin(angle)
    return torch.tensor([
        [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
        [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
    ])


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = torch.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.torch(gaussian_1D, gaussian_1D, indexing='xy')
    distance = (x**2 + y**2)** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = torch.exp(-(distance - mu)**2 / (2 * sigma**2))
    gaussian_2D = gaussian_2D / (2 * torch.pi * sigma**2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / torch.sum(gaussian_2D)
    return gaussian_2D

def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x**2 + y**2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D

def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
        kernel_angle = kernel_angle * is_diag   # because of the interpolation
        thin_kernels.append(kernel_angle)

    return thin_kernels

class CannyFilter(nn.Module):
    def __init__(self, k_gaussian=3, mu=0, sigma=1, k_sobel=3):
        super().__init__()

        # low_threshold = nn.Parameter(torch.rand(1))
        # high_threshold = nn.Parameter(torch.rand(1))

        # gaussian
        # gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        # self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         # out_channels=1,
                                         # kernel_size=k_gaussian,
                                         # padding=k_gaussian // 2,
                                         # bias=False)
        # self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)
        self.gaussian_filter = GaussianBlur(k_gaussian)

        # sobel
        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight = nn.Parameter(torch.from_numpy(sobel_2D), requires_grad=False)

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight = nn.Parameter(torch.from_numpy(sobel_2D.T), requires_grad=False)


        # thin
        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight.data[:, 0] = torch.from_numpy(directional_kernels)
        self.directional_filter.weight.requires_grad = False

        # hysteresis
        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight = nn.Parameter(torch.from_numpy(hysteresis), requires_grad=False)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(DEVICE)
        grad_x = torch.zeros((B, 1, H, W)).to(DEVICE)
        grad_y = torch.zeros((B, 1, H, W)).to(DEVICE)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(DEVICE)
        grad_orientation = torch.zeros((B, 1, H, W)).to(DEVICE)

        # gaussian
        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges
        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x**2 + grad_y**2)**0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges
        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds
        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        # return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges
        return grad_magnitude
