import flax.linen as nn
import jax.numpy as jnp


class InputLayer(nn.Module):
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return inputs


class OutputLayer(nn.Module):
    @nn.compact
    def __call__(self, outputs: jnp.ndarray) -> jnp.ndarray:
        return outputs
