from dataclasses import dataclass, field, asdict
from fleiadex.config.global_config import GlobalConfig

from omegaconf import OmegaConf


@dataclass
class Hyperparams:
    learning_rate: float | str = "optax.warmup_cosine_decay_schedule(1e-5, 1e-5, 3000, 80000, 1e-7)"
    batch_size: int = 10
    step: int = 149999
    kld_weight: float = 0
    disc_weight: float = 1e-2
    discriminator_start_after: int = 10000
    gradient_clipping: float = 1.

    save_ckpt: bool = True
    save_discriminator: bool = True
    ckpt_freq: int = 1000
    eval_freq: int = 1
    save_comparison: bool = True
    sample_size: int = 10
    save_sample: bool = True


@dataclass
class DataSpec:
    _target_: str = "fleiadex.data_module.FleiadexDataLoader"
    image_size: int = 128
    auto_normalisation: bool = True
    rescale_min: float | None = None
    rescale_max: float | None = None
    data_dir: str = "./src/fleiadex/exp_data/satel_array_202312bandopt00_clear.npy"
    validation_size: float = 0.1
    batch_size: int = Hyperparams.batch_size
    target_layout: str = "h w c"
    image_channels: int = 4


@dataclass
class VAENNSpec:
    encoder_spatial_downsample_schedule: tuple[int] = (2, 2, 2)
    encoder_channel_schedule: tuple[int] = (128, 256, 512)
    encoder_resnet_depth_schedule: tuple[int] = (2, 2, 2, 2)
    encoder_use_attention: bool = True
    encoder_attention_heads: int = 4
    encoder_attention_use_qkv_bias: bool = False
    encoder_attention_use_dropout: bool = True
    encoder_attention_dropout_rate: float = 0.1
    encoder_use_memory_efficient_attention = True
    encoder_post_attention_resnet_depth: int = 1
    encoder_latents_channels: int = 1
    encoder_conv_kernel_sizes: tuple[int] = (3, 3)
    encoder_down_sample_activation: str = "silu"
    encoder_post_attention_activation: str = "silu"
    encoder_final_activation: str = "silu"

    decoder_latent_channels: int = 1
    decoder_spatial_upsample_schedule: tuple[int] = (2, 2, 2)
    decoder_channel_schedule: tuple[int] = (512, 256, 128)
    decoder_resnet_depth_schedule: tuple[int] = (2, 2, 2)
    decoder_use_attention: bool = True
    decoder_attention_heads: int = 4
    decoder_attention_use_qkv_bias: bool = False
    decoder_attention_use_dropout: bool = True
    decoder_attention_dropout_rate: float = 0.1
    decoder_up_sampler_type: str = "resize"
    decoder_use_memory_efficient_attention: bool = True
    decoder_pre_output_resnet_depth: int = 1
    decoder_use_final_linear_projection: bool = False
    decoder_reconstruction_channels: int = DataSpec.image_channels
    decoder_conv_kernel_sizes: tuple[int] = (3, 3)
    decoder_up_sample_activation: str = "silu"
    decoder_pre_output_activation: str = "silu"
    decoder_final_activation: str = "silu"


@dataclass
class VAEConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: VAENNSpec = field(default_factory=VAENNSpec)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
