# simpler encodings attention
import torch.nn as nn
import torch.nn.functional as F
import torch

from math import ceil

from .attn_modules import SAttn_TBlock

from .enformer_modules import Attentions
from .config_enformer import AttentionConfig
# segment-wise attention


ncat = lambda tensors: torch.cat(tensors,dim=-2)



class SW_Attn(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.segment_length = config.segment_length
        self.n_segments = config.n_segments

        attn_config = AttentionConfig(
            dim = config.dim,
            heads = config.heads,
            dim_key = config.attn_dim_key,
            dim_value = config.dim // config.heads,
            dropout = config.attn_dropout,
            pos_dropout = config.pos_dropout,
            num_rel_pos_features = config.dim // config.heads,
            input_length = self.segment_length,
        )
        self.segment_wise = SAttn_TBlock(config,Attentions[config.attn_embed](attn_config))
        

    def forward(self,x):
        # perform self-attention on each sl-length segment
        # note that if !(n%n_segments==0), then the last segment
        # will be <sl in length
        b, n, d = x.shape
        n_segments = ceil(n/self.segment_length)
        sl = self.segment_length
        x = ncat(
            [
                self.segment_wise(
                    x[:,sl*i:sl*(i+1),:]
                )
                for i in range(n_segments)
            ],
        )
        return x

class CS_Attn(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.segment_length = config.segment_length
        self.n_segments = config.input_length // config.segment_length
        self.representative_length = config.representative_length
        input_length = config.n_segments * self.representative_length
        attn_config = AttentionConfig(
            dim = config.dim,
            heads = config.heads,
            dim_key = config.attn_dim_key,
            dim_value = config.dim // config.heads,
            dropout = config.attn_dropout,
            pos_dropout = config.pos_dropout,
            num_rel_pos_features = config.dim // config.heads,
            input_length = input_length,
        )

        self.attn = SAttn_TBlock(config,Attentions[config.attn_embed](attn_config))
    def forward(self,x):
        # collect the first `rl` positions in each segment
        sl = self.segment_length
        rl = self.representative_length
        representatives = ncat(
            [
                x[ : , sl*i:sl*i+rl , : ]
                for i in range(self.n_segments)
            ]
        )
        representatives = self.attn(representatives)
        
        # insert newly computed representatives 
        return ncat(
            [
                ncat(
                    [
                        representatives[ : , rl*i:rl*(i+1), :],
                        x[ : , sl*i+rl:sl*(i+1) , : ],
                    ],
                )
                for i in range(self.n_segments)
            ],
        )

class H_TBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attn = nn.Sequential(
            *[
                SW_Attn(config),
                CS_Attn(config),
            ]
        )
    def forward(self,x):
        return self.attn(x)


