from absl import app
import hydra
from hydra.core.config_store import ConfigStore
import tensorflow as tf
import os

from src.pleiades.trainers.diffuser_trainer_with_constants import Trainer
from config.ldm_config import LDMConfig


cs = ConfigStore.instance()
cs.store(name='ldm_config', node=LDMConfig)


@hydra.main(version_base=None, config_name="ldm_config")
def execute(config: LDMConfig) -> None:
    os.environ['CUDA_VISIBLE_DEVICE'] = config.global_config.use_which_gpus
    trainer = Trainer(config)
    trainer.load_vae_from("/home/arezy/Desktop/ProjectPleiades/training_scripts/vae/outputs/"
                          "2024-07-27/16-43-15/results/vae_ckpt")
    trainer.train()


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    tf.config.experimental.set_visible_devices([], 'GPU')

    execute()


if __name__ == '__main__':
    app.run(main)