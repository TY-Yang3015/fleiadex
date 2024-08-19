import hydra
import os, sys, logging

from fleiadex.config import VAEConfig


def execute_vae(config_dir: str) -> None:
    """
    execute the variational autoencoder with the specified YAML config file.

    :param config_dir: str, the config file path. must end with ``.yaml``.

    """

    try:
        sys.argv = sys.argv.remove('-f')
        pass
    except ValueError:
        pass
    except AttributeError:
        pass

    if sys.argv is None:
        sys.argv = [os.path.basename(__file__)]

    if not config_dir.endswith('.yaml'):
        raise ValueError("config_dir must end with .yaml.")

    if config_dir.startswith('~') or config_dir.startswith('./'):
        logging.basicConfig(level=logging.INFO)
        logging.warning('relative directory may cause issue for hydra (the config management library) in some '
                        'use cases, e.g. jupyter notebook. the current working directory is '
                        f'{os.getcwd()}.')

    @hydra.main(version_base=None, config_path=config_dir.replace(config_dir.split('/')[-1], ''),
                config_name=config_dir.split('/')[-1].replace('.yaml', ''))
    def execute_prediff_vae(config: VAEConfig) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.global_config.use_which_gpus
        os.environ["HYDRA_FULL_ERROR"] = config.global_config.hydra_full_error
        from fleiadex.trainers import get_vae_trainer

        trainer = get_vae_trainer(config)
        trainer.train()

    execute_prediff_vae()
