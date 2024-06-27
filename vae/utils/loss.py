import tensorflow as tf
import jax
import jax.numpy as jnp
import flax.linen as nn


@jax.vmap
def sse(epsilon_true, epsilon_pred):
    return jnp.sum(jnp.square(epsilon_true - epsilon_pred))


@jax.vmap
def kl_divergence(mean, log_var):
    return -0.5 * jnp.sum(1 + log_var - jnp.square(mean) - jnp.exp(log_var))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(
        labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
    )
