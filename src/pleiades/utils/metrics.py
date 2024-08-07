import tensorflow as tf
import jax.numpy as jnp

def ssim(origin: jnp.ndarray, generated: jnp.ndarray):
    max_val = jnp.maximum(origin.max(), generated.max())
    min_val = jnp.minimum(origin.min(), generated.min())
    return tf.reduce_mean(tf.image.ssim(origin, generated,
                                        max_val=max_val - min_val))