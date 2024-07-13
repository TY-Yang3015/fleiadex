import flax.linen as nn
import jax.numpy as jnp


class DownSampler_BatchNorm(nn.Module):
    leaky_relu_slope: float = 0.2
    conv_kernel_size: tuple[int, int] = (4, 4)
    spatial_downsample_factor: int = 2

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.leaky_relu(x, self.leaky_relu_slope)
        x = nn.Conv(features=x.shape[-1] * self.spatial_downsample_factor,
                    kernel_size=self.conv_kernel_size,
                    padding='SAME',
                    strides=(self.spatial_downsample_factor, self.spatial_downsample_factor),
                    kernel=nn.initializers.kaiming_normal(),
                    use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x

