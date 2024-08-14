from dataclasses import dataclass, field
from config.backbone_specs import *
from config.global_config import GlobalConfig


@dataclass
class DataSpec:
    _target_: str = "src.pleiades.data_module.DataLoader"
    pre_encoded: bool = True
    dataset_dir: str = "/home/arezy/Desktop/ProjectPleiades/src/pleiades/exp_data"
    batch_size: int = 8
    mean: tuple[float] = (264.5569, 249.6994, 228.8998, 242.5480)
    std: tuple[float] = (25.1830, 18.3450, 7.5322, 13.1226)
    input_image_size: tuple[int] = (64, 64)
    output_image_size: tuple[int] = (64, 64)
    image_channels: int = 4


@dataclass
class Hyperparams:
    learning_rate: float | str = "optax.cosine_decay_schedule(1e-5, 80000, 1e-7)"
    batch_size: int = 2
    diffusion_time_steps: int = 1000
    step: int = 100000
    save_ckpt: bool = True
    ckpt_freq: int = 2000
    save_prediction: bool = True
    pred_channel: int = 0


@dataclass
class ConditionalDDPMConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: UNet4DConv2DSpec = field(default_factory=UNet4DConv2DSpec)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
