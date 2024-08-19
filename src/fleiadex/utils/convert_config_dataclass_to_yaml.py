from omegaconf import OmegaConf

from fleiadex.config import VAEConfig, ConditionalDDPMConfig, LDMConfig


def convert_config_dataclass_to_yaml(config_type: str, save_dir: str) -> None:
    if config_type == "vae".casefold():
        config = VAEConfig()
    elif config_type == "conditional_ddpm".casefold():
        config = ConditionalDDPMConfig()
    elif config_type == "ldm".casefold():
        config = LDMConfig()
    else:
        raise ValueError(f"Config type {config_type} not supported. choose from "
                         f"'vae', 'conditional_ddpm', 'ldm'.'")

    if not save_dir.endswith('.yaml'):
        save_dir += f'{config_type}_config.yaml'

    config = OmegaConf.structured(config)
    OmegaConf.save(config, save_dir)
