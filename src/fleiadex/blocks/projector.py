import flax.linen as nn
import jax.numpy as jnp
import jax


class Projector(nn.Module):
    """
    projecting the input channel into the desire channel numbers

    :cvar drop_rate: dropout rate of the nn.Dropout layer.
    :cvar output_channels: the desired channel numbers. default 256.
    :cvar group: number of groups for the nn.GroupNorm layer. default 32.

    """

    drop_rate: float
    output_channels: int = 256
    group: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        """
        :param x: input tensor of shape ``(batch, ..., input_channels)``.
        :return: the projected input with shape ``(batch, ..., output_channels)``.
        """
        x = nn.GroupNorm(
            num_groups=self.group if x.shape[-1] % self.group == 0 else x.shape[-1],
            group_size=None,
        )(x)
        x = nn.silu(x)
        x = nn.Conv(
            features=self.output_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            name="projector_conv1",
        )(x)
        x = nn.GroupNorm(
            num_groups=self.group if x.shape[-1] % self.group == 0 else x.shape[-1],
            group_size=None,
        )(x)
        x = nn.silu(x)
        x = nn.Dropout(
            rate=self.drop_rate, deterministic=not train, name="projector_dropout"
        )(x)
        x = nn.Conv(
            features=self.output_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            name="projector_conv2",
        )(x)
        return x


# rng = jax.random.PRNGKey(0)
# print(Projector(0.1, 256).tabulate(rng, jnp.ones((10, 5, 16, 16, 5)), train=False))
