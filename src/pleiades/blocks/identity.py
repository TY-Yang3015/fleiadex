import flax.linen as nn
import jax.numpy as jnp


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x, train=None):
        return x