# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training and evaluation logic."""

from absl import logging
from flax import linen as nn
import models
from flax.training import train_state
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds
from src.utils.load_dataset import load_dataset
from src.utils.loss import kl_divergence, sse
from src.utils.save_image import save_image

@jax.jit
def compute_metrics(recon_x, x, mean, logvar):
    sse_loss = sse(recon_x, x).mean()
    kld_loss = kl_divergence(mean, logvar).mean() * 100.
    return {'sse': sse_loss, 'kld': kld_loss, 'loss': sse_loss + kld_loss}



def train_step(state, batch, z_rng, latents):
    @jax.jit
    def loss_fn(params):
        recon_x, mean, logvar = models.model(latents).apply(
            {'params': params}, batch, z_rng
        )

        sse_loss = sse(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = sse_loss + kld_loss * 100.
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


def eval_f(params, images, z, z_rng, latents):
    def eval_model(vae):
        recon_images, mean, logvar = vae(images, z_rng)
        comparison = jnp.concatenate([
            images[:8].reshape(-1, 128, 128, 3),
            recon_images[:8].reshape(-1, 128, 128, 3),
        ])

        generate_images = vae.generate(z)
        generate_images = generate_images.reshape(-1, 128, 128, 3)
        metrics = compute_metrics(recon_images, images, mean, logvar)
        return metrics, comparison, generate_images

    return nn.apply(eval_model, models.model(latents))({'params': params})


def train_and_evaluate(config: ml_collections.ConfigDict):
    """Train and evaulate pipeline."""
    rng = random.key(0)
    rng, key = random.split(rng)

    logging.info('Initializing dataset.')
    train_ds, test_ds = load_dataset(config)

    logging.info('Initializing model.')
    init_data = jnp.ones((config.hyperparams.batch_size, config.data_spec.image_size, config.data_spec.image_size, 3),
                         jnp.float32)
    params = models.model(config.latents).init(key, init_data, rng)['params']

    state = train_state.TrainState.create(
        apply_fn=models.model(config.latents).apply,
        params=params,
        tx=optax.adam(config.hyperparams.learning_rate),
    )

    rng, z_key, eval_rng = random.split(rng, 3)
    z = random.normal(z_key, (64, config.latents))

    for epoch in range(config.hyperparams.epochs):

        for _ in range(2):
            batch = next(train_ds)
            rng, key = random.split(rng)
            state = train_step(state, batch, key, config.latents)

        metrics, comparison, sample = eval_f(
            state.params, next(test_ds), z, eval_rng, config.latents
        )
        save_image(
            comparison, f'./results/reconstruction_{epoch}.png', nrow=8
        )
        save_image(sample, f'./results/sample_{epoch}.png', nrow=8)

        print(
            'eval epoch: {}, loss: {:.4f}, SSE: {:.4f}, KLD: {:.4f}'.format(
                epoch + 1, metrics['loss'], metrics['sse'], metrics['kld']
            )
        )
