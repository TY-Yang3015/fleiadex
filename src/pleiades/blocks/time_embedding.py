from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from einops import rearrange, repeat


class TimeEmbeddingInit(nn.Module):
    """
    dense layers to learn the time embedding.

    :cvar time_embedding_channels: the number of channels of the time embedding.
    """
    time_embedding_channels: int

    @nn.compact
    def __call__(self, time_stamps: jnp.ndarray,
                 max_period: int = 10000,
                 is_repeat: bool = False) -> jnp.ndarray:
        time_embedding = self.timestep_sinusoidal_embedding(time_stamps,
                                                            self.time_embedding_channels,
                                                            max_period,
                                                            is_repeat)
        time_embedding = nn.Dense(self.time_embedding_channels,
                                  kernel_init=nn.initializers.kaiming_normal(),
                                  bias_init=nn.initializers.zeros_init(),
                                  name=f'time_embedding_dense1')(time_embedding)
        time_embedding = nn.silu(time_embedding)
        time_embedding = nn.Dense(self.time_embedding_channels,
                                  kernel_init=nn.initializers.kaiming_normal(),
                                  bias_init=nn.initializers.zeros_init(),
                                  name=f'time_embedding_dense2')(time_embedding)
        return time_embedding

    def timestep_sinusoidal_embedding(self, time_stamp: jnp.ndarray,
                                      sinusoidal_embedding_dimensions: int,
                                      max_period: int,
                                      is_repeat: bool) -> jnp.ndarray:
        if not is_repeat:
            half = sinusoidal_embedding_dimensions // 2
            frequencies = jnp.linspace(0, (half - 1) / half, half)
            frequencies *= -jnp.log(max_period)
            frequencies = jnp.exp(frequencies)
            arguments = time_stamp.reshape(-1, 1) * frequencies.reshape(1, -1)
            embedding = jnp.concatenate([jnp.cos(arguments), jnp.sin(arguments)], axis=-1)
            if sinusoidal_embedding_dimensions % 2:
                embedding = jnp.concatenate([jnp.sin(arguments), jnp.zeros_like(embedding[:, :1])],
                                            axis=-1)
        else:
            embedding = repeat(time_stamp, 'b -> b d', d=sinusoidal_embedding_dimensions)

        return embedding


# print(TimeEmbeddingInit(128).tabulate(jax.random.PRNGKey(42),jnp.array([2, 3, 4, 5]),
#  console_kwargs={'width':150}))


class TimeEmbedding(nn.Module):
    """
    the time embedding layer with residual connection at the end.

    :cvar output_channels: the number of channels of the time embedding.
    :cvar dropout: the dropout probability. default: 0.1
    """
    output_channels: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        """
        :param x: the data tensor. should have the shape ``(batch, ..., channels)``
        :param emb: the time embedding tensor from ``TimeEmbeddingInit``
         or the previous ``TimeEmbedding`` layer. should have the shape ``(batch, channels)``
        :return: the data tensor with time embedding encoded.
        """

        h = nn.GroupNorm(num_groups=32 if self.output_channels % 32 == 0 else self.output_channels)(x)
        h = nn.silu(h)
        h = nn.Conv(features=self.output_channels,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=1)(h)

        emb = nn.silu(emb)
        emb = nn.Dense(features=self.output_channels)(emb)
        emb = rearrange(emb, 'b c -> b 1 1 1 c')
        h += emb

        h = nn.GroupNorm(num_groups=32 if self.output_channels % 32 == 0 else self.output_channels)(h)
        h = nn.silu(h)
        h = nn.Dropout(self.dropout, deterministic=not train)(h)
        h = nn.Conv(features=self.output_channels,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=1)(h)

        return x + h

#print(TimeEmbedding(128, 0.1).tabulate(jax.random.PRNGKey(1), jnp.ones((10, 5, 16,
#16, 128)), jnp.ones((10, 124)), True, console_kwargs={'width':150}))
