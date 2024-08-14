import flax.linen as nn
import jax.numpy as jnp
import jax
from fleiadex.utils import get_activation


class ResNetBlock(nn.Module):
    output_channels: int | None = None
    activation: str = "silu"
    group: int = 32

    @nn.compact
    def __call__(self, x):
        if self.output_channels is not None and x.shape[-1] != self.output_channels:
            residual = jax.image.resize(
                x,
                (x.shape[0], x.shape[1], x.shape[2], self.output_channels),
                method="nearest",
            )
        else:
            residual = x

        x = nn.GroupNorm(
            num_groups=self.group if x.shape[-1] % self.group == 0 else x.shape[-1],
            group_size=None,
        )(x)
        x = nn.Conv(
            features=x.shape[-1]
            if self.output_channels is None
            else self.output_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )(x)

        x = nn.GroupNorm(
            num_groups=self.group if x.shape[-1] % self.group == 0 else x.shape[-1],
            group_size=None,
        )(x)
        x = nn.Conv(
            features=x.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding="SAME"
        )(x)
        x = get_activation(self.activation)(x)
        return x + residual
