# simpler encodings attention
import torch.nn as nn
import torch.nn.functional as F
import torch

from math import ceil

from .attn_modules import SAttn_TBlock

from .enformer_modules import Attentions
from .config_enformer import AttentionConfig
# segment-wise attention

from einops import rearrange

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
        return self.segment_wise(x)
        

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
        rl = self.representative_length
        reps = x[:,:,:rl,:]
        reps = rearrange(reps,'b s n d -> b (s n) d')
        reps = self.attn(reps)
        reps = rearrange(reps,'b (s n) d -> b s n d',n=self.representative_length)
        x = torch.cat((reps,x[:,:,rl:,:]),dim=-2)
        return x

class H_TBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.segment_length = config.segment_length
        self.attn = nn.Sequential(
            *[
                SW_Attn(config),
                CS_Attn(config),
            ]
        )
    def forward(self,x):
        x = rearrange(x,'b (s n) d -> b s n d',n = self.segment_length)
        x = self.attn(x)
        x = rearrange(x,'b s n d -> b (s n) d')
        return x


