import flax.linen as nn
import jax.numpy as jnp
import jax


class FFNDropout(nn.Module):
    drop_rate: float

    @nn.compact
    def __call__(self, x, train):
        x = nn.Dropout(self.drop_rate, deterministic=not train)(x)
        return x


class PositionalFFN(nn.Module):
    input_channels: int
    hidden_size: int
    activation: str = "gelu"
    activation_dropout: float = 0.1
    dropout: float = 0.1

    pre_norm: bool = True
    gated_projection: bool = False

    def setup(self):
        self.dropout_layer = FFNDropout(self.dropout)
        self.activation_dropout_layer = FFNDropout(self.activation_dropout)

        self.ffn1 = nn.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.ffn1_gate = nn.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_init=nn.initializers.kaiming_normal(),
        )

        try:
            self.activation_function = eval("nn." + self.activation)
        except AttributeError:
            raise AttributeError(
                "please choose a valid activation function in flax.activation."
            )

        self.ffn2 = nn.Dense(self.input_channels, use_bias=True)
        self.layer_norm = nn.LayerNorm(1e-5, scale_init=nn.initializers.normal())

    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        residual = x
        if self.pre_norm:
            x = self.layer_norm(x)

        if self.gated_projection:
            out = self.activation_function(self.ffn1_gate(x)) * self.ffn1(x)
        else:
            out = self.activation_function(self.ffn1(x))

        out = self.activation_dropout_layer(out, not train)
        out = self.ffn2(out)
        out = self.dropout_layer(out, not train)
        out += residual

        if not self.pre_norm:
            out = self.layer_norm(out)

        return out


# rng = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
# print(PositionalFFN(256, 256*4).tabulate(rng, jnp.ones((10, 5, 16, 16, 256)), False, depth=1))
