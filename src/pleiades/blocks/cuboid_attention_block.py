import flax.linen as nn
import jax.numpy as jnp
from dataclasses import field
import jax

from src.pleiades.blocks.positional_ffn import PositionalFFN
from src.pleiades.blocks.cuboid_self_attention import CuboidSelfAttention
from config.transformer_attention_config import PatternFactory


def default_cuboid_sizes():
    return [(4, 4, 4), (4, 4, 4)]


def default_shift_sizes():
    return [(0, 0, 0), (2, 2, 2)]


def default_strategies():
    return [('d', 'd', 'd'), ('l', 'l', 'l')]


class CuboidAttentionBlock(nn.Module):
    '''
    a

    takes


    '''
    input_shape: tuple[int, int, int, int]
    attention_heads: int
    attention_pattern: str = 'axial'
    padding_type: str = 'auto'
    qkv_bias: bool = False
    attention_dropout: float = 0.
    projection_dropout: float = 0.
    ffn_dropout: float = 0.
    ffn_activation: str = 'gelu'
    gated_projection: bool = False
    use_inter_attention_ffn: bool = True
    use_relative_position: bool = True
    use_final_projection: bool = True

    def setup(self):
        self.input_channels = self.input_shape[-1]

        try:
            self.attention_pattern_spec = eval('PatternFactory(self.input_shape)'
                                               + f'.{self.attention_pattern}()')
        except AttributeError:
            raise ValueError(f'unknown attention pattern type. please choose '
                             f'from {str([method for method in dir(PatternFactory)
                                          if not method.startswith('_')]
                                         ).replace('[', '').replace(']', '')}.')

        self.cuboid_sizes, self.shift_sizes, self.strategies = self.attention_pattern_spec

        if len(self.cuboid_sizes[0]) == 0 or len(self.shift_sizes) == 0 or \
                len(self.strategies) == 0:
            raise ValueError("cuboid_sizes, shift_sizes and strategies cannot be empty.")

        if len(self.cuboid_sizes) != len(self.shift_sizes) != len(self.strategies):
            raise ValueError('the input cuboid_sizes and shift_sizes and strategies '
                             'must have the same length.')

        self.attention_blocks = len(self.cuboid_sizes)

        if self.use_inter_attention_ffn:
            self.ffn_list = [
                PositionalFFN(
                    input_channels=self.input_channels,
                    hidden_size=self.input_channels * 4,
                    activation=self.ffn_activation,
                    activation_dropout=self.ffn_dropout,
                    dropout=self.ffn_dropout
                ) for _ in range(self.attention_blocks)
            ]
        else:
            self.ffn_list = [
                PositionalFFN(
                    input_channels=self.input_channels,
                    hidden_size=self.input_channels * 4,
                    activation=self.ffn_activation,
                    activation_dropout=self.ffn_dropout,
                    dropout=self.ffn_dropout
                )]

        self.attention_list = [
            CuboidSelfAttention(
                attention_heads=self.attention_heads,
                input_channels=self.input_channels,
                cuboid_size=cuboid_size,
                shift_size=shift_size,
                strategy=strategy,
                padding_type=self.padding_type,
                qkv_bias=self.qkv_bias,
                attention_dropout_rate=self.attention_dropout,
                use_relative_position=self.use_relative_position,
                use_final_projector=self.use_final_projection,
                final_projection_dropout=self.projection_dropout,
            ) for cuboid_size, shift_size, strategy in
            zip(self.cuboid_sizes, self.shift_sizes, self.strategies)
        ]

    def __call__(self, x, train: bool):
        if self.use_inter_attention_ffn:
            for idx, (attention_block, ffn_layer) in enumerate(zip(self.attention_list, self.ffn_list)):
                x += attention_block(x, train=train)
                x = ffn_layer(x, train=train)
            return x
        else:
            for idx, attention_block in enumerate(self.attention_list):
                x += attention_block(x, train=train)
            x = self.ffn_list[0](x, train=train)
            return x


#rng = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
#print(CuboidAttentionBlock((5, 16, 16, 256), 4).tabulate(rng, jnp.ones((10, 5, 16, 16, 256)), False,
#                                                                depth=1, console_kwargs={'width': 150}))
