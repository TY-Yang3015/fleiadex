import flax.linen as nn
import jax.numpy as jnp
import jax
from src.pleiades.blocks.memory_efficient_attention import MemoryEfficientAttention
from einops import rearrange


class SelfAttention(nn.Module):
    """
    This is a light wrapper around `flax.linen.MultiHeadDotProductAttention` with a GroupNorm and an output projection.
    If the option ``use_memory_efficient_attention`` is set to ``True``, the memory efficient attention mechanism adapted from
    the hugging face ``diffuser`` module will be used.

    :cvar output_channels: number of projected output channels.
    :cvar num_heads: number of attention heads.
    :cvar use_memory_efficient_attention: whether to use memory efficient attention.
    :cvar group: number of groups used for GroupNorm.
    :cvar use_qkv_bias: whether to use bias in the QKV matrix.
    :cvar use_dropout: whether to use dropout in the attention layer.
    :cvar dropout_rate: dropout rate for the attention dropout, only used if use_dropout=True.

    """

    output_channels: int
    num_heads: int = 8
    use_memory_efficient_attention: bool = True
    group: int = 32
    use_qkv_bias: bool = False
    use_dropout: bool = True
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = nn.GroupNorm(
            num_groups=self.group if x.shape[-1] % self.group == 0 else x.shape[-1],
            group_size=None,
        )(x)
        shape = x.shape
        x = rearrange(x, "b w h c -> b (w h) c")

        if self.use_memory_efficient_attention:
            x = MemoryEfficientAttention(
                query_dim=shape[-1],
                heads=self.num_heads,
                dim_head=self.output_channels,
                dropout=self.dropout_rate if self.use_dropout else 0.0,
            )(x, deterministic=not train)
        else:
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=shape[-1],
                out_features=self.output_channels,
                dropout_rate=0 if self.use_dropout is False else self.dropout_rate,
                deterministic=not train,
                use_bias=self.use_qkv_bias,
            )(x)

        x = x.reshape(shape)
        x = nn.Dense(self.output_channels)(x)
        return x


# print(SelfAttention(512).tabulate(jax.random.PRNGKey(0), jnp.ones((5, 16, 16, 512)), False,\
#                                      depth=1, console_kwargs={'width':150}))
