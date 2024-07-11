from dataclasses import dataclass, field
from config.utils.vae_base import VAENNSpec


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
    dataset_dir: str = '../src/pleiades/exp_data/satel_array_202312bandopt00_clear.npy'
    validation_split: float = 0.2


@dataclass
class CNN2dSpec(VAENNSpec):
    nn_type: str = '2d_cnn'
    latents: int = 2048

    num_of_layers: int = 3
    stride: int = 2
    kernel_size: int = 3


@dataclass
class VAEConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: VAENNSpec = field(default_factory=CNN2dSpec)
