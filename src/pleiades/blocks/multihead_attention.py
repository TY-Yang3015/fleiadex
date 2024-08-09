import flax.linen as nn
import jax.numpy as jnp
import jax
from src.pleiades.blocks.memory_efficient_attention import MemoryEfficientAttention
from einops import rearrange


class MultiHeadCrossAttention(nn.Module):
    """
    This is a light wrapper around `flax.linen.MultiHeadDotProductAttention` with a GroupNorm and an output projection.

    :cvar output_channels: number of projected output channels.
    :cvar group: number of groups used for GroupNorm.
    :cvar use_qkv_bias: whether to use bias in the QKV matrix.
    :cvar use_dropout: whether to use dropout in the attention layer.
    :cvar dropout_rate: dropout rate for the attention dropout, only used if use_dropout=True.

    """

    output_channels: int
    num_heads: int = 4
    use_memory_efficient_attention: bool = False
    group: int = 32
    use_qkv_bias: bool = False
    use_dropout: bool = True
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool, context: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        x = nn.GroupNorm(
            num_groups=self.group if x.shape[-1] % self.group == 0 else x.shape[-1],
            group_size=None,
        )(x)
        shape = x.shape
        x = rearrange(x, "b w h c -> b (w h) c")

        if context is not None:
            context = rearrange(context, "b w h c -> b (w h) c")

        if self.use_memory_efficient_attention:
            x = MemoryEfficientAttention(
                query_dim=shape[-1],
                heads=self.num_heads,
                dim_head=self.output_channels,
                dropout=self.dropout_rate if self.use_dropout else 0.0,
            )(x, deterministic=not train, context=context)
        else:
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=shape[-1],
                out_features=self.output_channels,
                dropout_rate=0 if self.use_dropout is False else self.dropout_rate,
                deterministic=not train,
                use_bias=self.use_qkv_bias,
            )(x, inputs_k=context, inputs_v=context)

        x = x.reshape(shape)
        x = nn.Dense(self.output_channels)(x)
        return x


# print(MultiHeadCrossAttention(512).tabulate(jax.random.PRNGKey(0), jnp.ones((5, 16, 16, 512)),
#                                 False, jnp.ones((5, 16, 16, 512)),
#                                 depth=1, console_kwargs={'width':150}))
