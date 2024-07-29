from dataclasses import dataclass, field


@dataclass
class DataSpec:
    pre_encoded: bool = False
    image_size: int = 128
    image_channels: int = 4
    clip_min: float = 0.
    clip_max: float = 1
    dataset_dir: str = '../../src/pleiades/exp_data/satel_array_202312bandopt00_clear.npy'
    validation_split: float = 0.2
    condition_length: int = 3
    prediction_length: int = 2


@dataclass
class Hyperparams:
    learning_rate: float | str = "optax.cosine_decay_schedule(1e-5, 80000, 1e-7)"
    batch_size: int = 2
    diffusion_time_steps: int = 1000
    step: int = 100000

    save_ckpt: bool = True
    ckpt_freq: int = 2000
    save_prediction: bool = True


@dataclass
class EarthformerSpec:
    sample_input_shape: tuple[int, int, int, int] = (DataSpec.prediction_length, 32, 32, 3)
    cond_input_shape: tuple[int, int, int, int] = (DataSpec.condition_length, 32, 32, 3)
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
    use_attention_final_projection: bool = False
    attention_with_final_projection: bool = True
    time_embedding_channels_multiplier: int = 4
    time_embedding_dropout_rate: float = 0.1


@dataclass
class LDMConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: EarthformerSpec = field(default_factory=EarthformerSpec)
