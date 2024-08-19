from absl import app
import hydra
from hydra.core.config_store import ConfigStore
import os

from fleiadex.config import LDMConfig
from fleiadex.config import ConditionalDDPMConfig


cs = ConfigStore.instance()
cs.store(name="diffusion_config", node=ConditionalDDPMConfig)


@hydra.main(version_base=None, config_name="diffusion_config")
def execute(config: LDMConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.global_config.use_which_gpus
    os.environ["HYDRA_FULL_ERROR"] = config.global_config.hydra_full_error
    from src.fleiadex.trainers import get_diffuser_trainer

    trainer = get_diffuser_trainer(config)
    trainer.train(force_visualisation=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    execute()


if __name__ == "__main__":
    app.run(main)
