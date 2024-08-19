from dataclasses import dataclass, field
from fleiadex.config.global_config import GlobalConfig
from fleiadex.config.backbone_specs import UNet4DConv2DSpec


@dataclass
class DataSpec:
    _target_: str = "fleiadex.data_module.FleiadexDataLoader"
    pre_encoded: bool = True
    data_dir: str = "./src/fleiadex/exp_data/satel_array_202312bandopt00_clear.npy"
    validation_size: float = 0.1
    batch_size: int = 10
    fixed_normalisation_spec: tuple[list[float, ...], ...] | None = None
    auto_normalisation: bool = True
    output_image_size: int = 64
    image_channels: int = 4
    pre_split: bool = False
    condition_length: int = 3
    sample_length: int = image_channels - condition_length


@dataclass
class Hyperparams:
    learning_rate: float | str = "optax.warmup_cosine_decay_schedule(1e-5, 1e-5, 3000, 80000, 1e-7)"
    batch_size: int = DataSpec.batch_size
    diffusion_time_steps: int = 1000
    step: int = 99999
    save_ckpt: bool = True
    ckpt_freq: int = 2000
    save_prediction: bool = True
    gradient_clipping: float = 1.0
    ema_decay: float = 0.999

    load_vae_dir: str | None = '/home/arezy/Desktop/fleiadex/outputs/2024-08-17/18-13-15/results/vae_ckpt'


@dataclass
class ConditionalDDPMConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: UNet4DConv2DSpec = field(default_factory=UNet4DConv2DSpec)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
