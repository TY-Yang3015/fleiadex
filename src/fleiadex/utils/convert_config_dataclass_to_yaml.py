from omegaconf import OmegaConf

from fleiadex.config import VAEConfig, ConditionalDDPMConfig, LDMConfig


def get_yaml_config(config_type: str, save_dir: str) -> None:
    """
    covert the pre-encoded config to YAML. you can change the config file
    based on the output template.

    :param config_type: str. choose from ``'vae'``, ``'ldm'`` or ``'conditional_ddpm'``.
    :param save_dir: str. path to save the YAML config file.

    """
    if config_type.casefold() == "vae":
        config = VAEConfig()
    elif config_type.casefold() == "conditional_ddpm":
        config = ConditionalDDPMConfig()
    elif config_type.casefold() == "ldm":
        config = LDMConfig()
    else:
        raise ValueError(f"config type {config_type} not supported. choose from "
                         f"'vae', 'conditional_ddpm', 'ldm'.'")

    if not save_dir.endswith('.yaml'):
        save_dir += f'{config_type.casefold()}_config.yaml'

    config = OmegaConf.structured(config)
    OmegaConf.save(config, save_dir)
