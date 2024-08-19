from dataclasses import dataclass, field


@dataclass
class GlobalConfig:
    use_which_gpus: str = "0"
    use_diffuser_backbone: str = "unet_4d_conv_2d"
    use_predictor_backbone: str = "unet"
    save_num_ckpts: int = 3
    compute_auxiliary_metric: bool = True
