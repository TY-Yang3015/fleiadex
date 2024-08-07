from dataclasses import dataclass

class PredictorConfig:
    num_samples: int = 10
    features: int = 128
    layers: int = 4
