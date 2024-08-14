from absl import logging
from flax.core import FrozenDict
from jax import random
import jax
from jax.lib import xla_bridge
from clu import platform
import tensorflow as tf

from omegaconf import OmegaConf


class Trainer:
    def __init__(self, config: VAEConfsig):
        tf.config.experimental.set_visible_devices([], "GPU")

        logging.info(f"JAX backend: {xla_bridge.get_backend().platform}")

        logging.info(f"JAX process: {jax.process_index() + 1} / {jax.process_count()}")
        logging.info(f"JAX local devices: {jax.local_devices()}")

        platform.work_unit().set_task_status(
            f"process_index: {jax.process_index()}, "
            f"process_count: {jax.process_count()}"
        )
        # convert to FrozenDict, the standard config container in jax
        self.config: FrozenDict = FrozenDict(OmegaConf.to_container(config))

        # initialise the random number generator keys
        latent_rng, self.eval_rng, self.dropout_rng, self.train_key = self._init_rng()
        # train_key is only used to split keys
        self.train_key, self.train_rng = random.split(self.train_key)

        # initialise dataset with custom pipelines
