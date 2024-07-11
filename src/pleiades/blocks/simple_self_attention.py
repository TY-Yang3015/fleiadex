import flax.linen as nn
import jax.numpy as jnp
import jax


class SelfAttention(nn.Module):
    output_channels: int
    group: int = 32
    attention_heads: int = 4
    use_qkv_bias: bool = False
    use_dropout: bool = True
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train):
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
