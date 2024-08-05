from dataclasses import dataclass


@dataclass
class EarthformerSpec:
    sample_input_shape: tuple[int, int, int, int] = (3, 32, 32, 3)
    cond_input_shape: tuple[int, int, int, int] = (2, 32, 32, 3)
    base_units: int = 128
    bottleneck_depth: int = 8
    spatial_downsample_factor: int = 2
    upsampler_kernel_size: int = 3
    block_attention_pattern: str = 'axial'
    padding_type: str = 'auto'
    unet_use_residual_connection: bool = True
    attention_heads: int = 4
    attention_dropout: float = 0.1
    projection_dropout: float = 0.1
    ffn_dropout: float = 0.1
    ffn_activation: str = 'gelu'
    gated_ffn: bool = False
    use_inter_attention_ffn: bool = False
    positional_embedding_type: str = 't+h+w'
    use_relative_position: bool = True
    use_attention_final_projection: bool = True
    time_embedding_channels_multiplier: int = 4
    time_embedding_dropout_rate: float = 0.1
