from dataclasses import dataclass, field
from backbone_specs import EarthformerSpec

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
class LDMConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    data_spec: DataSpec = field(default_factory=DataSpec)
    nn_spec: any = field(default_factory=EarthformerSpec)
