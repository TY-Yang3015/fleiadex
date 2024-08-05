from dataclasses import dataclass, field


@dataclass
class GlobalConfig:
    use_which_gpus: str = '0, 1'
    use_diffuser_backbone: str = 'vanilla2d'
