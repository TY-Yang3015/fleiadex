import flax.linen as nn
import jax.numpy as jnp
import jax


class ResNetBlock(nn.Module):
    group: int = 32

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.GroupNorm(num_groups=self.group if x.shape[-1] % self.group == 0 else x.shape[-1],
                         group_size=None)(x)
        x = nn.Conv(features=x.shape[-1],
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME')(x)
        x = nn.GroupNorm(num_groups=self.group if x.shape[-1] % self.group == 0 else x.shape[-1],
                         group_size=None)(x)
        x = nn.Conv(features=x.shape[-1],
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME')(x)
        x = nn.silu(x)
        return x + residual
