from absl import app
from absl import logging
from clu import platform
import jax
import hydra
from hydra.core.config_store import ConfigStore
import tensorflow as tf

from src.pleiades.diffuser.diffuser_trainer import Trainer
from config.ldm_config import LDMConfig
from jax.lib import xla_bridge


def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except Exception:
        return False


cs = ConfigStore.instance()
cs.store(name='ldm_config', node=LDMConfig)


@hydra.main(version_base=None, config_name="ldm_config")
def execute(config: LDMConfig) -> None:
    trainer = Trainer(config)
    trainer.load_vae_from("/home/arezy/Desktop/ProjectPleiades/training_scripts/vae/outputs/"
                          "2024-07-27/16-43-15/results/vae_ckpt")
    trainer.train()


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info(f'JAX backend: {xla_bridge.get_backend().platform}')

    if xla_bridge.get_backend().platform == 'gpu':
        if jax_has_gpu():
            pass
        else:
            logging.warning('JAX GPU not intialised properly.')

    logging.info(f'JAX process: {jax.process_index() + 1} / {jax.process_count()}')
    logging.info(f'JAX local devices: {jax.local_devices()}')

    platform.work_unit().set_task_status(
        f'process_index: {jax.process_index()}, '
        f'process_count: {jax.process_count()}'
    )

    execute()


if __name__ == '__main__':
    app.run(main)