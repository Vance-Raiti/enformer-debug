from .attn_modules import SAttn_TBlock
from .enformer_modules import Attention, Residual, ConvBlock, exponential_linspace_int, AttentionPool, TargetLengthCrop
from .enformer_modules import GELU, map_values, exists, ConvBlockLN, ConvBlockP
from einops.layers.torch import Rearrange
from einops import rearrange
from .config_enformer import EnformerConfig, AttentionConfig
import torch
import torch.nn as nn

from .h_attn import H_TBlock
from .henf_config import HEnformerConfig

from .h_attn_r import H_TBlock as H_TBlock_R

from torch.utils.checkpoint import checkpoint_sequential
from copy import copy

from putils import memusage,gac

from .enformer_modules import tnan


class Enformer(nn.Module):

    @staticmethod
    def from_hparams(**kwargs):
        return Enformer(HEnformerConfig(**kwargs))

    def __init__(
            self,
            config,
        ):
        
        
        super().__init__()
        

        self.config = config
        
        if config.dim == -1:
            config.dim = 1536//2**(7-config.num_downsamples)
            print(f'Hierarchical Enformer: dim is {config.dim}')
        config.key_dim = config.dim
        config.value_dim = config.dim
        print('henf using conv block:')
        if config.conv_patch: # patch eval with track_running_stats=False
            print('patched conv')
            conv_block = ConvBlockP
        elif config.batched:
            print('batch conv')
            conv_block = ConvBlock
        else:
            # ConvBlock uses BatchNorm, which has numerically unstable behavior on
            # batch_size = 1. This convblock just uses a layernorm
            print('layer norm conv')
            conv_block = ConvBlockLN

        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2

        # for transformer blocks/attention
        config.dim_key = config.dim // config.heads
        config.dim_value = config.dim
        config.kv_dim_in = config.dim
        config.q_dim_in = config.dim
        config.num_rel_pos_features = config.dim // config.heads
        # create stem

        print('h_enformer, half_dim:',half_dim)
        print()
        print()
        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )

        # create conv tower

        filter_list = exponential_linspace_int(half_dim, config.dim, num = (config.num_downsamples - 1), divisible_by = config.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))
        
        # The action of downsampling is actually important because it means that when we perform target_length_crop
        # that we're only cropping out about 50% of the information, so if we want to reduce the amount of downsampling
        # for the purposes of preserving positional information, perform the rest of the downsampling at the end
        self.conv_tower_prefix = nn.Sequential(*conv_layers[:config.prefix_downsamples]) 
        self.conv_tower_suffix = nn.Sequential(*conv_layers[config.prefix_downsamples:])
        # transformer
        attn_config = copy(config)
        attn_config.dim = filter_list[config.prefix_downsamples]
        attn_config.input_length = config.attn_input_length
        if config.rearrange:
            tblock = H_TBlock_R
        else:
            tblock = H_TBlock
        
        self.transformer = nn.Sequential(
            *[
                tblock(attn_config)
                for _ in range(config.depth)
            ]
        )
        
        
        # target cropping

        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(config.dim, config.dim*2, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(config.dropout_rate / 8),
            GELU()
        )

        # create trunk sequential module


        # create final heads for human and mouse

        self.add_heads(**config.output_heads)

        # use checkpointing on transformer trunk

        self.use_checkpointing = config.use_checkpointing
    def add_heads(self, **kwargs):
        self.output_heads = kwargs

        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim * 2, features),
            nn.Softplus()
        ), kwargs))

    @property
    def heads(self):
        return self._heads
    def forward(
        self,
        x,
        target = None,
        return_corr_coef = False,
        return_embeddings = False,
        return_only_embeddings = False,
        head = ['human'],
        target_length = None
    ):
        b, n, d = x.shape
        config = self.config
        if n != config.input_length:
            print(f'WARNNG: skipping input of length {n}')
            return None
        head = head[0]
        

        if isinstance(x, list):
            x = str_to_one_hot(x)

        elif x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)

        no_batch = x.ndim == 2
        if no_batch:
            x = rearrange(x, '... -> () ...')
        

        x = rearrange(x,'b n d -> b d n')
        x = self.stem(x)
        x = self.conv_tower_prefix(x)
        x = rearrange(x,'b d n -> b n d')
        
        x = checkpoint_sequential(
            functions = self.transformer,
            segments = config.depth-1,
            input = x,
        )
        x = rearrange(x,'b n d -> b d n')
        x = self.conv_tower_suffix(x)
        x = rearrange(x,'b d n -> b n d')

        x = self.crop_final(x)
        x = self.final_pointwise(x)
        if no_batch:
            x = rearrange(x, '() ... -> ...')
        if return_only_embeddings:
            return x

        out = map_values(lambda fn: fn(x), self._heads)

        if exists(head):
            assert head in self._heads, f'head {head} not found'
            return out[head]
        
        return out

enformer = gac(Enformer.from_hparams,HEnformerConfig.__annotations__)
