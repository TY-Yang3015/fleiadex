import shutil
from absl import logging
from functools import partial
import os
from flax import linen as nn
import flax
import jax
from jax import random
import jax.numpy as jnp
import optax

import hydra
from omegaconf import OmegaConf, DictConfig

import src.pleiades.vae.src.vae as models
from src.pleiades.utils import (load_dataset, save_image, save_model,
                                sse, kl_divergence, TrainStateWithDropout)


class Trainer(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config: flax.core.FrozenDict = flax.core.FrozenDict(OmegaConf.to_container(config))

        latent_rng, self.eval_rng, self.dropout_rng, self.train_key = self._init_rng()
        self.train_key, self.train_rng = random.split(self.train_key)

        latent_size = 1
        for down_factor in config['nn_spec']['encoder_spatial_downsample_schedule']:
            latent_size *= down_factor

        latent_size = int(config['data_spec']['image_size'] / latent_size)

        self.latent_sample: jax.Array = random.normal(latent_rng,
                                                      (config['hyperparams']['sample_size'],
                                                       latent_size, latent_size
                                                       , config['nn_spec']['decoder_latent_channels']))

        logging.info('initializing dataset.')
        self.train_ds, self.test_ds = load_dataset(self.config)
        self.save_dir = self._init_savedir()
        self.vae = models.VAE(**config['nn_spec'])

        if isinstance(self.config['hyperparams']['learning_rate'], float):
            self.optimiser = optax.adam(self.config['hyperparams']['learning_rate'])
        else:
            try:
                self.optimiser = optax.adam(eval(self.config['hyperparams']['learning_rate'], {"optax": optax}))
            except Exception as e:
                raise ValueError(f"unknown learning rate type: {e} \nplease follow optax syntax.")

    def _init_rng(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        init_key = random.key(42)
        latent_rng, eval_rng, dropout_rng, train_key = random.split(init_key, 4)
        return latent_rng, eval_rng, dropout_rng, train_key

    def _init_savedir(self) -> str:
        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        save_dir = str(os.path.join(save_dir, 'results'))
        os.makedirs(save_dir)
        return save_dir

    def _update_train_rng(self) -> None:
        self.train_key, self.train_rng = random.split(self.train_key)

    @partial(jax.jit, static_argnames='self')
    def _train_step(self, state, batch):
        grads = jax.grad(self._loss_fn)(state.params, batch)
        return state.apply_gradients(grads=grads)

    @partial(jax.jit, static_argnames='self')
    def _compute_metrics(self, recon_x: jax.Array, x: jax.Array, mean: jax.Array
                         , logvar: jax.Array) -> dict[str, jax.Array]:
        sse_loss = sse(recon_x, x).mean()
        kld_loss = kl_divergence(mean, logvar).mean() * self.config['hyperparams']['kld_weight']
        return {'sse': sse_loss, 'kld': kld_loss, 'loss': sse_loss + kld_loss}

    @partial(jax.jit, static_argnames='self')
    def _loss_fn(self, params, batch):
        recon_x, mean, logvar = self.vae.apply(
            {'params': params}, batch, self.train_key, True,
            rngs={'dropout': self.dropout_rng}
        )

        sse_loss = sse(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = sse_loss + kld_loss * self.config['hyperparams']['kld_weight']
        return loss

    def _evaluate_model(self, vae, images, latent_sample, eval_rng, train):
        size = self.config['data_spec']['image_size']
        channels = self.config['data_spec']['image_channels']

        recon_images, mean, logvar = vae(images, eval_rng, train)
        comparison = jnp.concatenate([
            images[:].reshape(-1, size, size, channels),
            recon_images[:].reshape(recon_images.shape[0], size, size, channels),
        ])

        generate_images = vae.generate(latent_sample)
        generate_images = generate_images.reshape(-1, size, size, channels)

        metrics = self._compute_metrics(recon_images, images, mean, logvar)
        return metrics, comparison, generate_images

    @partial(jax.jit, static_argnames='self')
    def _evaluate(self, params, images, latent_sample, eval_rng):
        return nn.apply(self._evaluate_model, self.vae)({"params": params}, images
                                                        , latent_sample, eval_rng, True,
                                                        rngs={'dropout': self.dropout_rng})

    def _save_output(self, save_dir, comparison, sample, epoch):
        if self.config['hyperparams']['save_comparison']:
            save_image(
                self.config, save_dir, comparison, f'/comparison_{int(epoch + 1)}.png', nrow=8
            )

        if self.config['hyperparams']['save_sample']:
            save_image(self.config, save_dir, sample, f'/sample_{int(epoch + 1)}.png', nrow=8)

    def _clear_result(self):
        directory = self.save_dir
        shutil.rmtree(directory)
        os.makedirs(directory)

    def train(self):

        logging.info('initializing model.')
        init_data = jnp.ones((self.config['hyperparams']['batch_size'],
                              self.config['data_spec']['image_size'],
                              self.config['data_spec']['image_size'],
                              self.config['data_spec']['image_channels']),
                             jnp.float32)

        rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(42)}
        params = self.vae.init(rngs, init_data, random.key(0), False)['params']

        state = TrainStateWithDropout.create(
            apply_fn=self.vae.apply,
            params=params,
            tx=self.optimiser,
            key=self.dropout_rng
        )

        for epoch in range(self.config['hyperparams']['epochs']):

            batch = next(self.train_ds)
            self._update_train_rng()
            state = self._train_step(state, batch)

            if self.config['hyperparams']['save_ckpt']:
                if (epoch % self.config['hyperparams']['ckpt_freq'] == 0) and (epoch != 0):
                    save_model(state, self.config, epoch)

            if (epoch % 100 == 0) and (epoch != 0):
                self._clear_result()

            metrics, comparison, sample = self._evaluate(
                state.params, next(self.test_ds), self.latent_sample,
                self.eval_rng)

            self._save_output(self.save_dir, comparison, sample, epoch)

            logging.info(
                'eval epoch: {}, loss: {:.4f}, sse: {:.4f}, kld: {:.4f}'.format(
                    epoch + 1, metrics['loss'], metrics['sse'], metrics['kld']
                )
            )
