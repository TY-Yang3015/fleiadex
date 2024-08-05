import flax.linen as nn
import jax.numpy as jnp
import jax
from einops import rearrange
from src.pleiades.blocks import DownSamplerBatchNorm


class DiscriminatorOutput(nn.Module):
    leaky_relu_negative_slope: float = 0.2
    kernel_sizes: tuple = (4, 4)

    @nn.compact
    def __call__(self, x):
        x = nn.leaky_relu(x, negative_slope=self.leaky_relu_negative_slope)
        x = nn.Conv(features=1,
                    kernel_size=self.kernel_sizes,
                    padding=1,
                    strides=(1, 1),
                    kernel_init=nn.initializers.kaiming_normal()
                    )(x)
        x = nn.avg_pool(x, (x.shape[1], x.shape[2]))
        x = rearrange(x, 'b h w c -> b (h w c)')
        return x


# print(DiscriminatorOutput().tabulate(jax.random.PRNGKey(0), jnp.ones((10, 15, 15, 512),
#                                                                     ), console_kwargs={'width': 150}))


class Discriminator(nn.Module):
    spatial_downsample_schedule: tuple[int] = (2, 2)
    base_channels: int = 64
    conv_kernel_sizes: tuple[int, int] = (4, 4)

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=self.base_channels,
                    kernel_size=self.conv_kernel_sizes,
                    strides=(self.spatial_downsample_schedule[0],
                             self.spatial_downsample_schedule[0]),
                    padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal(),
                    use_bias=False
                    )(x)

        for downsample_factor in self.spatial_downsample_schedule:
            x = DownSamplerBatchNorm(
                leaky_relu_slope=0.2,
                conv_kernel_size=self.conv_kernel_sizes,
                spatial_downsample_factor=downsample_factor,
                padding_type='SAME'
            )(x, train)

        x = DownSamplerBatchNorm(
            leaky_relu_slope=0.2,
            conv_kernel_size=self.conv_kernel_sizes,
            spatial_downsample_factor=2,
            padding_type=1
        )(x, train)

        x = DiscriminatorOutput()(x)
        x = nn.sigmoid(x)

        return x


# print(Discriminator().tabulate(jax.random.PRNGKey(0), jnp.ones((10, 64, 64, 1)), False,
#                               depth=1, console_kwargs={'width': 150}))
