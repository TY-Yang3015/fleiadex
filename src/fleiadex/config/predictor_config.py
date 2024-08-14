from dataclasses import dataclass, field
from fleiadex.config.global_config import GlobalConfig


@dataclass
class Hyperparams:
    learning_rate: float | str = "optax.cosine_decay_schedule(1e-4, 80000, 1e-7)"
    batch_size: int = 8
    step: int = 100000
    kld_weight: float = 0.0
    disc_weight: float = 1e-3
    discriminator_start_after: int = 50000

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
    auto_normalisation: bool = True
    rescale_min: float | None = None
    rescale_max: float | None = None
    dataset_dir: str = "/fleiadex/exp_data/satel_array_202312bandopt00_clear.npy"
    validation_split: float = 0.1


@dataclass
class NNSpec:
    features: int = 128
    layers: int = 4


@dataclass
class PredictorConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: NNSpec = field(default_factory=NNSpec)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
