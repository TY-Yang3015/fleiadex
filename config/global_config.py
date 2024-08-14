from dataclasses import dataclass, field


@dataclass
class GlobalConfig:
    use_which_gpus: str = "0, 1"
    use_diffuser_backbone: str = "unet_5d_conv_2d"
    use_predictor_backbone: str = "unet"
    save_num_ckpts: int = 3
