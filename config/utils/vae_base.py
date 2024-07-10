from dataclasses import dataclass
from typing import Optional


@dataclass
class VAENNSpec:
    nn_type: str
    latents: int

    num_of_layers: int
    stride: int
    kernel_size: int

    features: Optional[list] = None
    max_feature: Optional[int] = None
    decoder_input: Optional[int] = None
