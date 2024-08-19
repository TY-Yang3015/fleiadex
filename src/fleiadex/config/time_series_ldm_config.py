from dataclasses import dataclass, field
from fleiadex.config.global_config import GlobalConfig
from fleiadex.config.backbone_specs import Unet5DConv2DSpec


@dataclass
class Hyperparams:
    learning_rate: float | str = "optax.cosine_decay_schedule(1e-5, 80000, 1e-7)"
    batch_size: int = 5
    diffusion_time_steps: int = 1000
    step: int = 100000
    gradient_clipping: float = 1.0
    ema_decay: float = 0.999

    save_ckpt: bool = True
    ckpt_freq: int = 2000
    save_prediction: bool = True

    load_ckpt_dir: str | None = '/home/arezy/Desktop/fleiadex/outputs/2024-08-17/18-13-15/results/vae_ckpt'
    load_config: bool = True
    ckpt_step: int | None = None


@dataclass
class DataSpec:
    _target_: str = "fleiadex.data_module.FleiadexDataLoader"
    pre_encoded: bool = True
    data_dir: str = "/home/arezy/Desktop/fleiadex/src/fleiadex/exp_data/satel_1.npy"
    validation_size: float = 0.1
    batch_size: int = Hyperparams.batch_size
    fixed_normalisation_spec: tuple[list[float, ...], ...] | None = None
    auto_normalisation: bool = True
    output_image_size: int = 32
    image_channels: int = 4
    condition_length: int = 3
    prediction_length: int = 1
    pre_split: bool = False
    sequenced: bool = True
    sequence_length: int = condition_length + prediction_length


@dataclass
class LDMConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: Unet5DConv2DSpec = field(default_factory=Unet5DConv2DSpec)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
