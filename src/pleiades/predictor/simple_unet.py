import flax.linen as nn
import jax
import jax.numpy as jnp


class SimpleUNet(nn.Module):
    features: int
    layers: int
    dropout = 0.2
    ks: int = 4
    kernel_size: tuple[int, int] = (4, 4)
    window_shape: tuple[int, int] = (2, 2)
    strides: tuple[int, int] = (2, 2)
    input_dilation: tuple[int, int] = (2, 2)
    padding: tuple[tuple[int, int], tuple[int, int]] = ((1, ks - 1), (1, ks - 1))

    @nn.compact
    def __call__(self, x, training: bool):
        z = x
        zs = []
        # encoder
        for i in range(self.layers):
            z = nn.Conv(self.features * 2 ** i, kernel_size=(4, 4), kernel_init=jax.nn.initializers.glorot_uniform())(z)
            z = nn.relu(z)
            z = nn.Conv(self.features * 2 ** i, kernel_size=(4, 4), kernel_init=jax.nn.initializers.glorot_uniform())(z)
            z = nn.GroupNorm(num_groups=8)(z)
            z = nn.relu(z)
            zs.append(z)
            z = nn.max_pool(z, window_shape=(2, 2), strides=(2, 2))

        # bottleneck
        z = nn.Conv(self.features * 2 ** self.layers, kernel_size=(4, 4),
                    kernel_init=jax.nn.initializers.glorot_uniform())(z)
        z = nn.GroupNorm(num_groups=8)(z)
        z = nn.relu(z)
        z = nn.Conv(self.features * 2 ** self.layers, kernel_size=(4, 4),
                    kernel_init=jax.nn.initializers.glorot_uniform())(z)
        z = nn.GroupNorm(num_groups=8)(z)
        z = nn.relu(z)
        # print(z.shape)

        # decoder
        for i in range(self.layers):
            z = nn.Conv(self.features * 2 ** (self.layers - i), kernel_size=(4, 4),
                        kernel_init=jax.nn.initializers.glorot_uniform(), input_dilation=(2, 2),
                        padding=((1, self.ks - 1), (1, self.ks - 1)))(z)
            z = jnp.concatenate((zs[self.layers - i - 1], z), axis=3)
            z = nn.Conv(self.features * 2 ** (self.layers - i - 1), kernel_size=(4, 4),
                        kernel_init=jax.nn.initializers.glorot_uniform())(z)
            z = nn.GroupNorm(num_groups=8)(z)
            z = nn.relu(z)
            z = nn.Conv(self.features * 2 ** (self.layers - i - 1), kernel_size=(4, 4),
                        kernel_init=jax.nn.initializers.glorot_uniform())(z)
            z = nn.GroupNorm(num_groups=8)(z)
            z = nn.relu(z)
            # print(z.shape)

        # output layer
        z_conv = nn.Conv(1, kernel_size=(1, 1), kernel_init=jax.nn.initializers.glorot_uniform())(z)
        z_conv = nn.relu(z_conv)
        z_bin = nn.Dense(features=1)(z)
        z_bin = nn.sigmoid(z_bin)

        y_conv = z_conv
        y_bin = z_bin

        y = jnp.stack([y_bin, y_conv], axis=3)
        y = jnp.reshape(y, y.shape[:-1])
        return y