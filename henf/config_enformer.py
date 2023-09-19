

from putils import ConfigType
class EnformerConfig(ConfigType):
    model_type:str = "enformer"
    
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
    # Fanguri's Enformer
    transformer_with_linear: bool = False
    with_final_pointwise: bool = False
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # no reason not to
        self.output_heads = {
            'human': 5313,
            'mouse': 1643,
        }
class AttentionConfig(ConfigType):
    dim: int = 1536
    heads: int = 8
    num_rel_pos_features: int = 1536//8
    dim_key: int = 64 # dim that key matrix projects to. Will actually create one key for each head, so
    # total dim is dim_key * heads
    dim_value: int = 1536//8
    dropout: float = 0.0
    pos_dropout: float = 0.0
    pos_embed_dim: int = 64
    
    abs_pos_embed: bool = False # positional embeddings for the hierarchical enformer's CS Attention is going to be
    # difficult, so for now just use an absolute positional embedding
    input_length: int = None

