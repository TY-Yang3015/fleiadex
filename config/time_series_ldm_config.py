from dataclasses import dataclass, field
from config.backbone_specs import *
from config.global_config import GlobalConfig


@dataclass
class DataSpec:
    pre_encoded: bool = True
    image_size: int = 16
    image_channels: int = 4
    auto_normalisation: bool = True
    rescale_min: float | None = None
    rescale_max: float | None = None
    dataset_dir: str = "/home/arezy/Desktop/ProjectPleiades/src/pleiades/exp_data/satel_array_202312bandopt00_clear.npy"
    validation_split: float = 0.1
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
class LDMConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: Vanilla2DSpec = field(default_factory=Vanilla2DSpec)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
