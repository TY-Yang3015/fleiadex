import flax.linen as nn
import jax.numpy as jnp
import jax


class SelfAttention(nn.Module):
    """
    This is a light wrapper around `flax.linen.MultiheadAttention` with a GroupNorm and an output projection.

    :cvar output_channels: number of projected output channels.
    :cvar group: number of groups used for GroupNorm.
    :cvar attention_heads: number of attention heads.
    :cvar use_qkv_bias: whether to use bias in the QKV matrix.
    :cvar use_dropout: whether to use dropout in the attention layer.
    :cvar dropout_rate: dropout rate for the attention dropout, only used if use_dropout=True.

    """
    output_channels: int
    group: int = 32
    attention_heads: int = 4
    use_qkv_bias: bool = False
    use_dropout: bool = True
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = nn.GroupNorm(num_groups=self.group if x.shape[-1] % self.group == 0 else x.shape[-1],
                         group_size=None)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.attention_heads,
            qkv_features=x.shape[-1],
            out_features=self.output_channels,
            dropout_rate=0 if self.use_dropout is False else self.dropout_rate,
            deterministic=not train,
            use_bias=self.use_qkv_bias
        )(x)
        x = nn.Dense(self.output_channels)(x)
        return x

# print(SelfAttention(512).tabulate(jax.random.PRNGKey(0), jnp.ones((5, 16, 16, 512)), False,
#                                  depth=1, console_kwargs={'width':150}))
