import random
from math import pi, log
from functools import wraps
from typing import *

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

class CALNet(nn.Module):
    def __init__(
            self,
            *,
            n_modalities: int,
            channel_dims: List,
            num_spatial_axes: List,
            out_dims: int,
            depth: int = 3,
            num_freq_bands: int = 2,
            max_freq: float = 10.,
            l_c: int = 128,
            l_d: int = 128,
            x_heads: int = 8,
            l_heads: int = 8,
            cross_dim_head: int = 64,
            latent_dim_head: int = 64,
            attn_dropout: float = 0.,
            ff_dropout: float = 0.,
            weight_tie_layers: bool = False,
            fourier_encode_data: bool = True,
            self_per_cross_attn: int = 1,
    ):

        super().__init__()
        assert len(channel_dims) == len(num_spatial_axes), 'input channels and input axis must be of the same length'
        assert len(
            num_spatial_axes) == n_modalities, 'input axis must be of the same length as the number of modalities'

        self.input_axes = num_spatial_axes
        self.input_channels = channel_dims
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.modalities = n_modalities
        self.self_per_cross_attn = self_per_cross_attn
        self.n_modalities = n_modalities
        self.fourier_encode_data = fourier_encode_data



        self.sha = [5, 5, 1, 1]
        # get fourier channels and input dims for each modality
        fourier_channels = []
        input_dims = []
        for axis in num_spatial_axes:
            fourier_channels.append((axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0)
        for f_channels, i_channels in zip(fourier_channels, channel_dims):
            input_dims.append(f_channels + i_channels)

        # initialise shared latent bottleneck
        self.latents = nn.Parameter(torch.randn(l_c, l_d))
        self.y = []
        for i in range(self.n_modalities):
            # print(f'y: self.sha[i]:{self.sha[i]} channel_dims[i]:{channel_dims[i]}')
            self.y.append(nn.Parameter(torch.randn(self.sha[i], channel_dims[i])))


        # modality-specific attention layers
        funcs = []
        for m in range(n_modalities):
            funcs.append(lambda m=m: PreNorm(l_d, Attention(l_d, input_dims[m], heads=x_heads, dim_head=cross_dim_head,
                                                            dropout=attn_dropout), context_dim=input_dims[m]))
        cross_attn_funcs = tuple(map(cache_fn, tuple(funcs)))

        get_latent_attn = lambda: PreNorm(l_d,
                                          Attention(l_d, heads=l_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        get_cross_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout=ff_dropout))
        get_latent_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout=ff_dropout))

        get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])

        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(get_latent_attn(**cache_args, key=block_ind))
                self_attns.append(get_latent_ff(**cache_args, key=block_ind))

            cross_attn_layers = []
            for j in range(n_modalities):
                cross_attn_layers.append(cross_attn_funcs[j](**cache_args))
                cross_attn_layers.append(get_cross_ff(**cache_args))

            self.layers.append(nn.ModuleList(
                [*cross_attn_layers, self_attns])
            )



        self.cross_modal_attention = nn.ModuleList([])
        def get_cross_modal_module(query_dim, context_dim):
            cross_attn = PreNorm(
                query_dim,
                Attention(query_dim=query_dim,context_dim=context_dim,heads=8,dim_head=64,dropout=0.1),
                context_dim=context_dim)
            cross_ff = PreNorm(
                query_dim,
                FeedForward(query_dim, dropout=ff_dropout)
            )
            return nn.ModuleList([cross_attn, cross_ff])

        # 为每对模态创建带缓存的模块
        for i in range(n_modalities):
            for j in range(n_modalities):
                query_dim = self.input_channels[i]
                context_dim = self.input_channels[j]
                # 使用缓存机制（需在cache_fn中处理参数差异）
                cached_module = cache_fn(lambda i=i, j=j: get_cross_modal_module(query_dim, context_dim))()
                self.cross_modal_attention.append(cached_module)
    def forward(self,
                tensors: List[Union[torch.Tensor, None]],
                ):
        device = 'cuda'
        for i in range(len(tensors)):
            data = tensors[i]
            b, *axis, _, device, dtype = *data.shape, data.device, data.dtype

            data = rearrange(data, 'b ... d -> b (...) d')
            tensors[i] = data
            # print(f'tensors[i] shape: {tensors[i].shape}')

        x = repeat(self.latents, 'n d -> b n d', b=b)  # note: batch dim should be identical across modalities
        for layer_idx, layer in enumerate(self.layers):
            y = tensors.copy()
            for i in range(self.modalities):
                cross_attn = layer[i * 2]
                cross_ff = layer[(i * 2) + 1]
                for j in range(self.modalities):
                    if i == j:
                        continue
                    c1, c2 = self.cross_modal_attention[i*self.n_modalities+j]
                    tensors[i] = c1(tensors[i], context=tensors[j]) + tensors[i]
                    tensors[i] = c2(tensors[i]) + tensors[i]
                try:
                    x = cross_attn(x, context=tensors[i]) + x
                    x = cross_ff(x) + x
                except:
                    pass
                self_attn, self_ff = layer[-1]
                x = self_attn(x) + x
                x = self_ff(x) + x
        return x


# HELPERS/UTILS
"""
Helper class implementations based on: https://github.com/lucidrains/perceiver-pytorch
Helper class implementations based on: https://github.com/konst-int-i/healnet/tree/main/healnet
"""


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class SELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.selu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        activation = SELU()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            activation,
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def temperature_softmax(logits, temperature=1.0, dim=-1):
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=dim)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        # add leaky relu
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.LeakyReLU(negative_slope=1e-2)
        )
        self.attn_weights = None

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = temperature_softmax(sim, temperature=0.5, dim=-1)
        self.attn_weights = attn
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)






