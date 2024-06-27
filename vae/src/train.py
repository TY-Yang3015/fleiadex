from absl import logging
import os
from flax import linen as nn
from flax.training import train_state
import jax
from jax import random
import jax.numpy as jnp
import optax

import hydra
from omegaconf import OmegaConf, DictConfig

import vae.src.cnn_models as models
from vae.utils.load_dataset import load_dataset
from vae.utils.loss import kl_divergence, sse
from vae.utils.save_image import save_image
from vae.utils.save_model import save_model
from vae.utils.auto_designers import AutoDesigner


@jax.jit
def compute_metrics(recon_x, x, mean, logvar):
    sse_loss = sse(recon_x, x).mean()
    kld_loss = kl_divergence(mean, logvar).mean() * 100.
    return {'sse': sse_loss, 'kld': kld_loss, 'loss': sse_loss + kld_loss}


def train_step(state, batch, z_rng, config):
    @jax.jit
    def loss_fn(params):
        recon_x, mean, logvar = models.model(config).apply(
            {'params': params}, batch, z_rng
        )

        sse_loss = sse(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = sse_loss + kld_loss * 100.
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


def eval_f(params, images, z, z_rng, config):
    size = config.data_spec.image_size
    channels = config.data_spec.image_channels

    def eval_model(vae):
        recon_images, mean, logvar = vae(images, z_rng)
        comparison = jnp.concatenate([
            images[:8].reshape(-1, size, size, channels),
            recon_images[:8].reshape(-1, size, size, channels),
        ])

        generate_images = vae.generate(z)
        generate_images = generate_images.reshape(-1, size, size, channels)
        metrics = compute_metrics(recon_images, images, mean, logvar)
        return metrics, comparison, generate_images

    return nn.apply(eval_model, models.model(config))({'params': params})


@hydra.main(version_base=None, config_path="../../config/vae", config_name="config")
def train_and_evaluate(config: DictConfig):
    OmegaConf.set_readonly(config, True)

    rng = random.key(0)
    rng, key = random.split(rng)

    logging.info('initializing dataset.')
    train_ds, test_ds = load_dataset(config)

    logging.info('initializing model.')
    init_data = jnp.ones((config.hyperparams.batch_size,
                          config.data_spec.image_size,
                          config.data_spec.image_size,
                          config.data_spec.image_channels),
                         jnp.float32)
    designer = AutoDesigner(config)
    config = designer.design_config()
    params = models.model(config).init(key, init_data, rng)['params']

    state = train_state.TrainState.create(
        apply_fn=models.model(config).apply,
        params=params,
        tx=optax.adam(config.hyperparams.learning_rate),
    )

    rng, z_key, eval_rng = random.split(rng, 3)
    z = random.normal(z_key, (64, config.nn_spec.latents))

    for epoch in range(config.hyperparams.epochs):

        batch = next(train_ds)
        rng, key = random.split(rng)
        state = train_step(state, batch, key, config)

        if config.hyperparams.save_ckpt:
            if (epoch % config.hyperparams.ckpt_freq == 0) and (epoch != 0):
                save_model(state, config, epoch)

        metrics, comparison, sample = eval_f(
            state.params, next(test_ds), z, eval_rng, config
        )

        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        save_dir = str(os.path.join(save_dir, 'results'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        if config.hyperparams.save_comparison:
            save_image(
                comparison, save_dir + f'/reconstruction_{epoch}.png', nrow=8
            )

        if config.hyperparams.save_sample:
            save_image(sample, save_dir + f'/sample_{epoch}.png', nrow=8)

        logging.info(
            'eval epoch: {}, loss: {:.4f}, sse: {:.4f}, kld: {:.4f}'.format(
                epoch + 1, metrics['loss'], metrics['sse'], metrics['kld']
            )
        )
