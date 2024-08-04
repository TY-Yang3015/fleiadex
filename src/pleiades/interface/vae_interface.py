import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict
from jax import random
import optax

import orbax.checkpoint as ocp
import etils.epath as path

from src.pleiades.vae.prediff_vae.vae import VAE
from src.pleiades.utils.train_states.trainstate_with_dropout import TrainStateWithDropout


class VariationalAutoencoder:
    def __init__(self,
                 ckpt_dir: dir,
                 step: None | int = None):
        print('loading vae from %s', ckpt_dir)

        if isinstance(ckpt_dir, str):
            ckpt_dir = path.Path(ckpt_dir)
        else:
            raise ValueError('ckpt dir must be str.')

        mngr = ocp.CheckpointManager(
            ckpt_dir, item_names=('vae_state', 'config')
        )

        if step is not None:
            restored_config = mngr.restore(step,
                                           args=ocp.args.Composite(
                                               config=ocp.args.JsonRestore()
                                           ))
        else:
            restored_config = mngr.restore(mngr.latest_step(),
                                           args=ocp.args.Composite(
                                               config=ocp.args.JsonRestore()
                                           ))

            self.config = FrozenDict(restored_config['config'])

        self.vae = VAE(**self.config['nn_spec'])

        init_data = jnp.ones((self.config['hyperparams']['batch_size'],
                              self.config['data_spec']['image_size'],
                              self.config['data_spec']['image_size'],
                              self.config['data_spec']['image_channels']),
                             jnp.float32)

        rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(42)}
        params = self.vae.init(rngs, init_data, random.key(0), False)['params']

        self.vae_optimiser = self._get_optimiser(self.config)
        self.disc_optimiser = self._get_optimiser(self.config)

        vae_state = TrainStateWithDropout.create(
            apply_fn=self.vae.apply,
            params=params,
            tx=self.vae_optimiser,
            key=random.PRNGKey(42),
        )

        if step:
            restored = mngr.restore(step,
                                    args=ocp.args.Composite(
                                        vae_state=ocp.args.StandardRestore(vae_state)
                                    ))['vae_state']
        else:
            restored = mngr.restore(mngr.latest_step(),
                                    args=ocp.args.Composite(
                                        vae_state=ocp.args.StandardRestore(vae_state)
                                    ))['vae_state']

        self.vae = restored

        print('loading succeeded.')
        del mngr

    def _get_optimiser(self, config: FrozenDict) -> optax.GradientTransformation:
        if isinstance(config['hyperparams']['learning_rate'], float):
            return optax.adam(config['hyperparams']['learning_rate'])
        else:
            try:
                optimiser = optax.adam(eval(config['hyperparams']['learning_rate'], {"optax": optax}))
            except Exception as e:
                raise ValueError(f"unknown learning rate type: {e} \nplease follow optax syntax.")
            return optimiser

    def encode(self, image_data: jnp.ndarray) -> jnp.ndarray:

        if jnp.ndim(image_data) != 4:
            raise ValueError('image_data must be 4-dimensional.')

        image_data = self.vae.apply_fn({'params': self.vae.params},
                                       image_data, random.PRNGKey(42),
                                       rngs={'dropout': random.PRNGKey(0)}, method='encode')

        return image_data

    def decode(self, encoded_latent: jnp.ndarray) -> jnp.ndarray:

        if jnp.ndim(encoded_latent) != 4:
            raise ValueError('encoded_latent must be 4-dimensional.')

        decoded_image = self.vae.apply_fn({'params': self.vae.params},
                                          encoded_latent, method='decode')

        return decoded_image
