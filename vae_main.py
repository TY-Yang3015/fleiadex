import hydra
from hydra.core.config_store import ConfigStore
import os

from fleiadex.config import VAEConfig

cs = ConfigStore.instance()
cs.store(name="vae_config", node=VAEConfig)

from fleiadex.utils import convert_config_dataclass_to_yaml

#convert_config_dataclass_to_yaml('vae', './')


@hydra.main(version_base=None, config_path=None, config_name="vae_config")
def execute(config: VAEConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.global_config.use_which_gpus
    os.environ["HYDRA_FULL_ERROR"] = '0'
    from fleiadex.trainers import get_vae_trainer

    trainer = get_vae_trainer(config)
    # trainer.load_vae_from("/home/arezy/Desktop/fleiadex/training_scripts/vae/outputs/"
    #                      "2024-08-03/22-07-34/results/vae_ckpt", load_config=False)
    trainer.train()


if __name__ == "__main__":
    execute()
