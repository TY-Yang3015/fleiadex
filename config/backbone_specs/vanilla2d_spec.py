from dataclasses import dataclass

@dataclass
class Vanilla2DSpec:
    sample_input_shape: tuple[int, int, int, int] = (4, 16, 16, 4)
    cond_input_shape: tuple[int, int, int, int] = (1, 16, 16, 4)
    base_channels: int = 128
    spatial_down_sample_schedule: tuple[int] = (2, 2)
    merge_temporal_dim_and_image_channels: bool = False
    unet_use_residual: bool = True
    down_sampler_type: str = 'conv'
    down_sample_resnet_depth: int = 2
    down_sample_use_attention: bool = True
    down_sample_attention_heads: int = 4
    down_sample_attention_use_memory_efficient: bool = False
    down_sample_attention_use_dropout: bool = True
    down_sample_attention_dropout_rate: float = 0.1
    bottleneck_resnet_depth: int = 2
    bottleneck_use_attention: bool = True
    bottleneck_attention_heads: int = 4
    bottleneck_attention_use_memory_efficient: bool = True
    bottleneck_attention_use_dropout: bool = True
    bottleneck_attention_dropout_rate: float = 0.1
    up_sampler_type: str = 'interp2d'
    up_sample_resnet_depth: int = 2
    up_sample_use_attention: bool = True
    up_sample_attention_heads: int = 4
    up_sample_attention_use_memory_efficient: bool = True
    up_sample_attention_use_dropout: bool = True
    up_sample_attention_dropout_rate: float = 0.1