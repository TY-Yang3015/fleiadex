from absl import app
import hydra
from hydra.core.config_store import ConfigStore
import tensorflow as tf
import os
import src.pleiades.trainers.prediff_vae_trainer as train
from config.vae_config import VAEConfig


cs = ConfigStore.instance()
cs.store(name='vae_config', node=VAEConfig)


@hydra.main(version_base=None, config_name="vae_config")
def execute(config: VAEConfig) -> None:
    os.environ['CUDA_VISIBLE_DEVICE'] = config.global_config.use_which_gpus
    trainer = train.Trainer(config)
    #trainer.load_vae_from("/home/arezy/Desktop/ProjectPleiades/training_scripts/vae/outputs/"
    #                      "2024-08-03/22-07-34/results/vae_ckpt", load_config=False)
    trainer.train(auxiliary_metric=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    tf.config.experimental.set_visible_devices([], 'GPU')

    execute()


if __name__ == '__main__':
    app.run(main)