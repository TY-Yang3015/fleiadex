from absl import app
import hydra
from hydra.core.config_store import ConfigStore
import os

from fleiadex.config import LDMConfig
from fleiadex.config import ConditionalDDPMConfig


cs = ConfigStore.instance()
cs.store(name="ldm_config", node=ConditionalDDPMConfig)


@hydra.main(version_base=None, config_name="ldm_config")
def execute(config: LDMConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.global_config.use_which_gpus
    from src.fleiadex.trainers import get_diffuser_trainer

    trainer = get_diffuser_trainer(config)
    # trainer.load_vae_from("/home/arezy/Desktop/fleiadex/training_scripts/vae/outputs/"
    #                      "2024-07-27/16-43-15/results/vae_ckpt")
    trainer.train(True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    execute()


if __name__ == "__main__":
    app.run(main)
