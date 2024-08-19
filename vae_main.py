import hydra
from hydra.core.config_store import ConfigStore
import os

from fleiadex.config import VAEConfig

cs = ConfigStore.instance()
cs.store(name="vae_config", node=VAEConfig)


@hydra.main(version_base=None, config_path=None, config_name="vae_config")
def execute(config: VAEConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.global_config.use_which_gpus
    os.environ["HYDRA_FULL_ERROR"] = config.global_config.hydra_full_error
    from fleiadex.trainers import get_vae_trainer

    trainer = get_vae_trainer(config)
    trainer.train()


if __name__ == "__main__":
    execute()
