import flax.linen as nn
import jax.numpy as jnp
import jax.random


class DownSamplerBatchNorm(nn.Module):
    leaky_relu_slope: float = 0.2
    conv_kernel_size: tuple[int, int] = (4, 4)
    spatial_downsample_factor: int = 2
    padding_type: str | int = "SAME"

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.leaky_relu(x, self.leaky_relu_slope)

        if isinstance(self.padding_type, str):
            x = nn.Conv(
                features=x.shape[-1] * self.spatial_downsample_factor,
                kernel_size=self.conv_kernel_size,
                padding=self.padding_type,
                strides=(
                    self.spatial_downsample_factor,
                    self.spatial_downsample_factor,
                ),
                kernel_init=nn.initializers.kaiming_normal(),
                use_bias=False,
            )(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
        elif isinstance(self.padding_type, int):
            x = nn.Conv(
                features=x.shape[-1] * self.spatial_downsample_factor,
                kernel_size=self.conv_kernel_size,
                padding=self.padding_type,
                strides=(1, 1),
                kernel_init=nn.initializers.kaiming_normal(),
                use_bias=False,
            )(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            raise ValueError("padding type must be either str or int")

        return x


# print(DownSamplerBatchNorm().tabulate(jax.random.PRNGKey(0),
#                                      jnp.zeros((5, 16, 16, 256)), train=False,
#                                      console_kwargs={'width': 150}))
