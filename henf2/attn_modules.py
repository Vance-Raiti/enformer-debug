import torch
import torch.nn as nn

from .enformer_modules import Residual

class TransformerBlock(nn.Module):
    def __init__(
            self,
            config,
            attn,
        ):
        super().__init__()
        self.config = config
        dim = config.dim
        self.dim = dim
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.dim) 
            for x in ['q','k','v']
        ])

        
        self.attn = attn
        self.attn_dropout = nn.Dropout(config.dropout_rate)

        self.suffix = Residual(nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(config.dropout_rate)
        ))

    def forward(self,q,k,v):
        h = q
        q,k,v = [self.layer_norms[i](t) for i,t in enumerate([q,k,v])]
        y = self.attn(q,k,v)
        y = self.attn_dropout(y)
        y = y + h
        return self.suffix(y)

# two-sequence transformer block
class TS_TBlock(TransformerBlock):
    def __init__(self,config,*args):
        super().__init__(config,*args)
        if config.kv_in_dim is None:
            config.kv_in_dim = config.dim
        write_norm = nn.LayerNorm(config.dim)
        read_norm = nn.LayerNorm(config.kv_in_dim)

        self.layer_norms = nn.ModuleList(
            [read_norm,write_norm,write_norm]
        )

    def forward(self,read,write):
        return super().forward(
            q=write,
            k=read,
            v=read,
        )

class SAttn_TBlock(TransformerBlock):
    def __init__(self,config,attn):
        super().__init__(config,attn)
        self.layer_norm = nn.LayerNorm(config.dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.LayerNorm(config.dim),
            self.attn,
            self.attn_dropout,
        )
    def forward(self,x):
        h = x
        x = self.trunk(x)
        return self.suffix(x+h)


