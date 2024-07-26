import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange
import jax

from src.pleiades.blocks.utils import pad_input


class PatchMerge3D(nn.Module):
    input_channels: int
    down_sample_factors: tuple[int, int, int]
    output_channels: None | int = None
    padding_type: str = 'auto'

    def setup(self):
        self.output_channels_auto = self.output_channels
        if self.output_channels is None:
            self.output_channels_auto = (self.input_channels *
                                         self.down_sample_factors[0] * self.down_sample_factors[1] * self.down_sample_factors[2])

        self.reduction_layer = nn.Dense(self.output_channels_auto, use_bias=False,
                                        kernel_init=nn.initializers.kaiming_normal())
        self.norm_layer = nn.LayerNorm(scale_init=nn.initializers.normal())

    def get_output_shape(self, input_shape):
        t, h, w, c = input_shape

        pad_t = (self.down_sample_factors[0] - t % self.down_sample_factors[0]) % self.down_sample_factors[0]
        pad_h = (self.down_sample_factors[1] - h % self.down_sample_factors[1]) % self.down_sample_factors[1]
        pad_w = (self.down_sample_factors[2] - w % self.down_sample_factors[2]) % self.down_sample_factors[2]

        return ((t + pad_t) // self.down_sample_factors[0], (h + pad_h) // self.down_sample_factors[1],
                (w + pad_w) // self.down_sample_factors[2]), self.output_channels_auto

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, t, h, w, c = x.shape

        pad_t = (self.down_sample_factors[0] - t % self.down_sample_factors[0]) % self.down_sample_factors[0]
        pad_h = (self.down_sample_factors[1] - h % self.down_sample_factors[1]) % self.down_sample_factors[1]
        pad_w = (self.down_sample_factors[2] - w % self.down_sample_factors[2]) % self.down_sample_factors[2]

        t += pad_t
        h += pad_h
        w += pad_w
        x = pad_input(x, pad_t, pad_h, pad_w, self.padding_type)

        x = x.reshape(batch_size,
                      t // self.down_sample_factors[0], self.down_sample_factors[0],
                      h // self.down_sample_factors[1], self.down_sample_factors[1],
                      w // self.down_sample_factors[2], self.down_sample_factors[2], c)
        x = rearrange(x, 'a b c d e f g h -> a b d f c e g h')
        x = x.reshape(batch_size, t // self.down_sample_factors[0], h // self.down_sample_factors[1],
                      w // self.down_sample_factors[2], self.down_sample_factors[0] * self.down_sample_factors[1] * self.down_sample_factors[2] * c)
        x = self.norm_layer(x)
        x = self.reduction_layer(x)

        return x


#rng = jax.random.PRNGKey(0)
#print(PatchMerge3D(256, (1, 2, 2)).tabulate(rng, jnp.ones((10, 5, 16, 16, 256)),
#                                            console_kwargs={'width':150}))
