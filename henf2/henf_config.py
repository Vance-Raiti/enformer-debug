from .config_enformer import EnformerConfig
from math import ceil

class HEnformerConfig(EnformerConfig):
     
    # original enformer configs
    dim: int = 1536
    depth: int = 11
    heads: int = 8
    output_heads: dict = dict(human = 5313, mouse= 1643)
    target_length: int = 896
    attn_dim_key: int = 64
    dropout_rate: float = 0.4
    attn_dropout: float = 0.05
    pos_dropout: int = 0.01
    use_checkpointing: bool = False
    use_convnext: bool = False
    num_downsamples: int = 7    # genetic sequence is downsampled 2 ** 7 == 128x in default Enformer - can be changed for higher resolution
    dim_divisible_by: int = 128
    batched: bool = False
    conv_patch: bool = False
   
    
    
    input_length: int = 196608
    segment_length: int = 512
    representative_length: int = 32
    prefix_downsamples: int = 5
    rearrange: bool = True
    attn_embed = 'enformer'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.num_downsamples <= 16, self.num_downsamples
        assert self.num_downsamples > self.prefix_downsamples, f'{self.num_downsamples} > {self.prefix_downsamples}'
        n = self.input_length // 2**(self.prefix_downsamples+1)
        self.n_segments = ceil(n // self.segment_length)
        self.attn_input_length = n

