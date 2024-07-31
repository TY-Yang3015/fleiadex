from dataclasses import dataclass, field


@dataclass
class Hyperparams:
    learning_rate: float | str = "optax.cosine_decay_schedule(1e-4, 80000, 1e-7)"
    batch_size: int = 7
    step: int = 100000
    kld_weight: float = 1e-3
    disc_weight: float = 1e-3
    discriminator_start_after: int = 2000

    save_ckpt: bool = True
    save_discriminator: bool = True
    ckpt_freq: int = 1000
    save_comparison: bool = True
    sample_size: int = 10
    save_sample: bool = True


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
    encoder_spatial_downsample_schedule: tuple[int] = (2, 2, 2)
    encoder_channel_schedule: tuple[int] = (128, 256, 512)
    encoder_resnet_depth_schedule: tuple[int] = (2, 2, 2, 2)
    encoder_attention_use_qkv_bias: bool = True
    encoder_attention_use_dropout: bool = True
    encoder_attention_dropout_rate: float = 0.2
    encoder_use_memory_efficient_attention = True
    encoder_post_attention_resnet_depth: int = 2
    encoder_latents_channels: int = 4
    encoder_conv_kernel_sizes: tuple[int] = (3, 3)

    decoder_latent_channels: int = 4
    decoder_spatial_upsample_schedule: tuple[int] = (2, 2, 2)
    decoder_channel_schedule: tuple[int] = (512, 256, 128)
    decoder_resnet_depth_schedule: tuple[int] = (3, 3, 3)
    decoder_attention_use_qkv_bias: bool = True
    decoder_attention_use_dropout: bool = True
    decoder_attention_dropout_rate: float = 0.2
    decoder_use_memory_efficient_attention: bool = True
    decoder_pre_output_resnet_depth: int = 3
    decoder_reconstruction_channels: int = 4
    decoder_conv_kernel_sizes: tuple[int] = (3, 3)


@dataclass
class VAEConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: VAENNSpec = field(default_factory=VAENNSpec)
