import shutil

import jax.sharding
from absl import logging
from functools import partial
import os
from flax import linen as nn
from flax.core import FrozenDict
from jax import random, jit, grad
import jax.numpy as jnp
import einops
import optax

import orbax.checkpoint as ocp
import etils.epath as path

import hydra
from omegaconf import OmegaConf

import src.pleiades.vae.src.vae as models
from src.pleiades.utils import (load_dataset_for_diffusion, save_image, mse,
                                TrainStateWithDropout, DiffusorTrainState)
from src.pleiades.diffuser.ddpm_core import DDPMCore
from config.ldm_config import LDMConfig


class Trainer:
    def __init__(self, config: LDMConfig):

        # convert to FrozenDict, the standard config container in jax
        self.config: FrozenDict = FrozenDict(OmegaConf.to_container(config))

        # initialise the random number generator keys
        self.const_rng, self.eval_rng, self.dropout_rng, self.train_key = self._init_rng()
        # train_key is only used to split keys
        self.train_key, self.train_rng = random.split(self.train_key)

        # the diffusion core
        self.diffusor = DDPMCore(
            unet_type='earthformer',
            config=self.config,
            diffusion_time_steps=self.config['hyperparams']['diffusion_time_steps']
        )

        logging.info('initializing dataset.')
        self.train_ds, self.test_ds = load_dataset_for_diffusion(self.config)

        # initialise the save directory
        self.save_dir = self._init_savedir()

        # optimiser with exponential moving average
        self.optimiser = self._get_optimiser(self.config)
        self.optimiser = optax.chain(
            optax.ema(0.99),
            self.optimiser,
        )

        # indicator of vae availability
        self.__vae_ready__ = False
        self.vae = None

        # time frame length
        self.temporal_length = self.config['data_spec']['condition_length'] + self.config['data_spec'][
            'prediction_length']

    def _init_rng(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        init_key = random.PRNGKey(42)
        const_rng, eval_rng, dropout_rng, train_key = random.split(init_key, 4)
        return const_rng, eval_rng, dropout_rng, train_key

    def _init_savedir(self) -> str:
        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        save_dir = str(os.path.join(save_dir, 'results'))
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/predictions')
        return save_dir

    def _get_optimiser(self, config: FrozenDict) -> optax.GradientTransformation:
        if isinstance(config['hyperparams']['learning_rate'], float):
            return optax.adam(config['hyperparams']['learning_rate'])
        else:
            try:
                optimiser = optax.adam(eval(config['hyperparams']['learning_rate'], {"optax": optax}))
            except Exception as e:
                raise ValueError(f"unknown learning rate type: {e} \nplease follow optax syntax.")
            return optimiser

    def _update_train_rng(self) -> None:
        self.train_key, self.train_rng = random.split(self.train_key)

    def _get_next_train_batch(self):
        single = next(self.train_ds)
        if len(single) != self.temporal_length:
            single = next(self.train_ds)
        single = self.vae.apply_fn({'params': self.vae.params},
                                   single, self.eval_rng,
                                   rngs={'dropout': self.dropout_rng}, method='encode')
        batch = jnp.expand_dims(single, 0)

        for _ in range(self.config['hyperparams']['batch_size'] - 1):
            single = next(self.train_ds)
            if len(single) != self.temporal_length:
                single = next(self.train_ds)
            single = self.vae.apply_fn({'params': self.vae.params},
                                       single, self.eval_rng,
                                       rngs={'dropout': self.dropout_rng}, method='encode')
            single = jnp.expand_dims(single, 0)
            batch = jnp.append(batch, single, axis=0)

        return batch

    def _get_next_train_batch_pre_encoded(self):
        single = next(self.train_ds)
        if len(single) != self.temporal_length:
            single = next(self.train_ds)
        batch = jnp.expand_dims(single, 0)

        for _ in range(self.config['hyperparams']['batch_size'] - 1):
            single = next(self.train_ds)
            if len(single) != self.temporal_length:
                single = next(self.train_ds)
            single = jnp.expand_dims(single, 0)
            batch = jnp.append(batch, single, axis=0)

        return batch

    def _get_next_test_batch(self):
        single = next(self.test_ds)
        if len(single) != self.temporal_length:
            single = next(self.test_ds)
        single = self.vae.apply_fn({'params': self.vae.params},
                                   single, self.eval_rng,
                                   rngs={'dropout': self.dropout_rng}, method='encode')
        batch = jnp.expand_dims(single, 0)

        for _ in range(self.config['hyperparams']['batch_size'] - 1):
            single = next(self.test_ds)
            if len(single) != self.temporal_length:
                single = next(self.test_ds)
            single = self.vae.apply_fn({'params': self.vae.params},
                                       single, self.eval_rng,
                                       rngs={'dropout': self.dropout_rng}, method='encode')
            single = jnp.expand_dims(single, 0)
            batch = jnp.append(batch, single, axis=0)

        return batch

    def _get_next_test_batch_pre_coded(self):
        single = next(self.test_ds)
        if len(single) != self.temporal_length:
            single = next(self.test_ds)
        batch = jnp.expand_dims(single, 0)

        for _ in range(self.config['hyperparams']['batch_size'] - 1):
            single = next(self.test_ds)
            if len(single) != self.temporal_length:
                single = next(self.test_ds)
            single = jnp.expand_dims(single, 0)
            batch = jnp.append(batch, single, axis=0)

        return batch

    # @partial(jax.jit, static_argnames=('self',))
    def _train_step(self, state, batch):

        def loss_fn(params):
            pred_noise, true_noise = state.apply_fn(
                {'params': params, 'constants': state.consts},
                batch, self.train_rng, True,
                rngs={'dropout': self.dropout_rng, 'constants': self.const_rng}
            )

            mse_loss = mse(pred_noise, true_noise).mean()

            return mse_loss

        grads = jax.grad(loss_fn)(state.params)
        #updates, state = self.optimiser.update(grads, state, state.params)
        #state.params = optax.apply_updates(state.params, updates)
        state = state.apply_gradients(grads=grads)

        return state

    def load_vae_from(self, ckpt_dir):
        logging.info('loading vae from %s', ckpt_dir)

        if isinstance(ckpt_dir, str):
            ckpt_dir = path.Path(ckpt_dir)
        else:
            raise ValueError('checkpoint directory must be a string.')

        mngr = ocp.CheckpointManager(
            ckpt_dir, item_names=('vae_state', 'config')
        )

        restored_config = mngr.restore(mngr.latest_step(),
                                       args=ocp.args.Composite(
                                           config=ocp.args.JsonRestore()
                                       ))

        vae_config = FrozenDict(restored_config['config'])
        vae = models.VAE(**vae_config['nn_spec'])

        init_data = jnp.ones((vae_config['hyperparams']['batch_size'],
                              vae_config['data_spec']['image_size'],
                              vae_config['data_spec']['image_size'],
                              vae_config['data_spec']['image_channels']),
                             jnp.float32)

        rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(42)}
        params = vae.init(rngs, init_data, random.key(0), False)['params']

        vae_state = TrainStateWithDropout.create(
            apply_fn=vae.apply,
            params=params,
            tx=self._get_optimiser(vae_config),
            key=self.dropout_rng
        )

        restored = mngr.restore(mngr.latest_step(),
                                args=ocp.args.Composite(
                                    vae_state=ocp.args.StandardRestore(vae_state)
                                ))['vae_state']

        logging.info('loading succeeded.')

        self.__vae_ready__ = True
        self.vae = restored
        del mngr

    # @partial(jax.jit, static_argnums=0)
    def _evaluate(self, params, consts, test_batch):

        def compute_metric(pred, true) -> dict[str, jnp.ndarray]:
            mse_loss = mse(pred, true).mean()
            return {'mse': mse_loss, 'loss': mse_loss}

        def evaluate(diffuser):
            predictions = diffuser.generate_prediction(test_batch[:, :self.config['data_spec']['condition_length']],
                                                       self.eval_rng)

            predictions = jnp.concatenate([test_batch[:, :self.config['data_spec']['condition_length']], predictions],
                                          axis=1)

            predictions = einops.rearrange(predictions, 'b t w h c -> (b t) w h c')
            predictions = self.vae.apply_fn({'params': self.vae.params},
                                            predictions, method='decode')

            pred, true = diffuser.apply({'params': params, 'constants': consts},
                                        test_batch, self.eval_rng, False,
                                        rngs={'dropout': self.dropout_rng,
                                              'constants': self.const_rng}
                                        )

            metrics = compute_metric(pred, true)
            return metrics, predictions

        return nn.apply(evaluate, self.diffusor)({'params': params, 'constants': consts},
                                                 rngs={'dropout': self.dropout_rng,
                                                       'constants': self.const_rng})

    # @partial(jax.jit, static_argnums=0)
    def _evaluate_pre_encoded(self, params, consts, test_batch):
        def compute_metric(pred, true) -> dict[str, jnp.ndarray]:
            mse_loss = mse(pred, true).mean()
            return {'mse': mse_loss, 'loss': mse_loss}

        def evaluate(diffuser):
            predictions = None

            pred, true = diffuser.apply({'params': params, 'constants': consts},
                                        test_batch, self.eval_rng, False,
                                        rngs={'dropout': self.dropout_rng,
                                              'constants': self.const_rng}
                                        )

            metrics = compute_metric(pred, true)
            return metrics, predictions

        return nn.apply(evaluate, self.diffusor)({'params': params, 'constants': consts},
                                                 rngs={'dropout': self.dropout_rng,
                                                       'constants': self.const_rng})

    def _save_output(self, save_dir, prediction, step):
        if self.config['hyperparams']['save_prediction']:
            save_image(
                self.config, save_dir + '/predictions',
                prediction, f'/prediction_{int(step + 1)}.png',
                nrow=self.temporal_length
            )

    def _clear_result(self):
        directory = self.save_dir + '/predictions'
        shutil.rmtree(directory)
        os.makedirs(directory)

    def train(self):
        if self.config['data_spec']['pre_encoded'] is True:
            if self.__vae_ready__ is True:
                logging.warning('since the data is pre-encoded, the loaded VAE will be ignored.')
                del self.vae
            logging.warning('visualisation will be disabled.')

            logging.info('initializing model.')
            init_data = jnp.ones((self.config['hyperparams']['batch_size'],
                                  self.temporal_length,
                                  self.config['nn_spec']['sample_input_shape'][1],
                                  self.config['nn_spec']['sample_input_shape'][2],
                                  self.config['nn_spec']['sample_input_shape'][3]),
                                 jnp.float32)

            rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(42),
                    'constants': self.const_rng}
            collections = self.diffusor.init(rngs, init_data, self.train_rng, False)
            params = collections['params']
            consts = collections['constants']

            state = DiffusorTrainState.create(
                apply_fn=self.diffusor.apply,
                params=params,
                tx=self.optimiser,
                key=self.dropout_rng,
                consts=consts
            )

            save_path = ocp.test_utils.erase_and_create_empty(os.path.abspath(self.save_dir + '/ckpt'))
            save_options = ocp.CheckpointManagerOptions(max_to_keep=5,
                                                        save_interval_steps=self.config['hyperparams']['ckpt_freq'],
                                                        )

            if self.config['hyperparams']['save_ckpt']:
                diffusor_mngr = ocp.CheckpointManager(
                    save_path, options=save_options, item_names=('diffusor_state', 'config')
                )

            for step in range(1, self.config['hyperparams']['step'] + 1):

                batch = self._get_next_train_batch_pre_encoded()
                self._update_train_rng()

                state = self._train_step(state, batch)

                if self.config['hyperparams']['save_ckpt']:
                    diffusor_mngr.save(step, args=ocp.args.Composite(
                        diffusor_state=ocp.args.StandardSave(state),
                        config=ocp.args.JsonSave(self.config.unfreeze())
                    ))

                current_test = self._get_next_test_batch_pre_coded()

                metrics, prediction = self._evaluate_pre_encoded(
                    state.params, state.consts, current_test)

                logging.info(
                    'step: {}, loss: {:.4f}, mse: {:.4f}'.format(
                        step + 1, metrics['loss'], metrics['mse']
                    )
                )

            diffusor_mngr.wait_until_finished()

        else:
            if self.__vae_ready__ is False:
                raise ValueError('vae is not loaded. please run load_vae_from() method first.')

            logging.info('initializing model.')
            init_data = jnp.ones((self.config['hyperparams']['batch_size'],
                                  self.temporal_length,
                                  self.config['nn_spec']['sample_input_shape'][1],
                                  self.config['nn_spec']['sample_input_shape'][2],
                                  self.config['nn_spec']['sample_input_shape'][3]),
                                 jnp.float32)

            rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(42),
                    'constants': self.const_rng}
            collections = self.diffusor.init(rngs, init_data, self.train_rng, False)
            params = collections['params']
            consts = collections['constants']

            state = DiffusorTrainState.create(
                apply_fn=self.diffusor.apply,
                params=params,
                tx=self.optimiser,
                key=self.dropout_rng,
                consts=consts
            )

            save_path = ocp.test_utils.erase_and_create_empty(os.path.abspath(self.save_dir + '/ckpt'))
            save_options = ocp.CheckpointManagerOptions(max_to_keep=5,
                                                        save_interval_steps=self.config['hyperparams']['ckpt_freq'],
                                                        )

            if self.config['hyperparams']['save_ckpt']:
                diffusor_mngr = ocp.CheckpointManager(
                    save_path, options=save_options, item_names=('diffusor_state', 'config')
                )

            for step in range(1, self.config['hyperparams']['step'] + 1):

                batch = self._get_next_train_batch()
                self._update_train_rng()

                state = self._train_step(state, batch)

                if self.config['hyperparams']['save_ckpt']:
                    diffusor_mngr.save(step, args=ocp.args.Composite(
                        diffusor_state=ocp.args.StandardSave(state),
                        config=ocp.args.JsonSave(self.config.unfreeze())
                    ))

                current_test = self._get_next_test_batch()

                metrics, prediction = self._evaluate(
                    state.params, state.consts, current_test)

                self._save_output(self.save_dir, prediction, step)

                if (step % 1000 == 0) and (step != 0):
                    self._clear_result()

                logging.info(
                    'step: {}, loss: {:.4f}, mse: {:.4f}'.format(
                        step + 1, metrics['loss'], metrics['mse']
                    )
                )

            diffusor_mngr.wait_until_finished()
