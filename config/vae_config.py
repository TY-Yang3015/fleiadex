from dataclasses import dataclass, field


@dataclass
class Hyperparams:
    learning_rate: float | str = "optax.exponential_decay(\
        1e-4, 6000, 1e-2, 2000)"
    batch_size: int = 15
    epochs: int = 100000
    kld_weight: float = 0

    save_ckpt: bool = True
    ckpt_freq: int = 30000
    save_comparison: bool = True
    sample_size: int = 1
    save_sample: bool = False


@dataclass
class DataSpec:
    image_size: int = 128
    image_channels: int = 4
    clip_min: float = 0.
    clip_max: float = 1
    dataset_dir: str = '../../src/pleiades/exp_data/satel_array_202312bandopt00_clear.npy'
    validation_split: float = 0.2


@dataclass
class VAENNSpec:
    encoder_spatial_downsample_schedule: tuple[int] = (2, 2)
    encoder_channel_schedule: tuple[int] = (128, 256, 512)
    encoder_resnet_depth_schedule: tuple[int] = (2, 2, 2)
    encoder_attention_heads: int = 4
    encoder_attention_use_qkv_bias: bool = False
    encoder_attention_use_dropout: bool = True
    encoder_attention_dropout_rate: float = 0.1
    encoder_post_attention_resnet_depth: int = 2
    encoder_latents_channels: int = 3
    encoder_conv_kernel_sizes: tuple[int] = (3, 3)

    decoder_latent_channels: int = 3
    decoder_spatial_upsample_schedule: tuple[int] = (2, 2)
    decoder_channel_schedule: tuple[int] = (512, 256, 128)
    decoder_resnet_depth_schedule: tuple[int] = (2, 2, 2)
    decoder_attention_heads: int = 4
    decoder_attention_use_qkv_bias: bool = False
    decoder_attention_use_dropout: bool = True
    decoder_attention_dropout_rate: float = 0.1
    decoder_pre_output_resnet_depth: int = 3
    decoder_reconstruction_channels: int = 1
    decoder_conv_kernel_sizes: tuple[int] = (3, 3)


@dataclass
class VAEConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: VAENNSpec = field(default_factory=VAENNSpec)
