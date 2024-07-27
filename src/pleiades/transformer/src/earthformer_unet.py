import flax.linen as nn
import jax.numpy as jnp
import jax
from dataclasses import field

from src.pleiades.blocks import (ObservationalMask, Projector, TimeEmbeddingInit,
                                 TimeEmbedding, PositionalEmbedding, CuboidAttentionBlock,
                                 PatchMerge3D, UpSampler)
from src.pleiades.transformer.src.default_factories import DefaultFactory
from src.pleiades.errors import DimensionMismatchError, StructureError


class EarthformerUNet(nn.Module):
    sample_input_shape: tuple[int, int, int, int]
    cond_input_shape: tuple[int, int, int, int]
    base_units: int = 128
    io_depth: list[int] \
        = field(default_factory=DefaultFactory.default_depth)
    bottleneck_depth: int = 8
    spatial_downsample_factor: int = 2
    upsampler_kernel_size: int = 3
    block_attention_pattern: str = 'axial'
    block_cuboid_sizes: list[tuple[int, int, int]] \
        = field(default_factory=DefaultFactory.default_block_cuboid_sizes)
    block_cuboid_strategies: list[tuple[str, str, str]] \
        = field(default_factory=DefaultFactory.default_block_cuboid_strategy)
    padding_type: str = 'auto'
    unet_use_residual_connection: bool = True
    attention_heads: int = 4
    attention_dropout: float = 0.
    projection_dropout: float = 0.
    ffn_dropout: float = 0.
    ffn_activation: str = 'gelu'
    gated_ffn: bool = False
    use_inter_attention_ffn: bool = False
    positional_embedding_type: str = 't+h+w'
    use_relative_position: bool = True
    use_attention_final_projection: bool = False
    attention_with_final_projection: bool = True
    time_embedding_channels_multiplier: int = 4
    time_embedding_dropout_rate: float = 0.

    def setup(self) -> None:
        # self.current_shape = jnp.array([self.sample_input_shape[0] + self.cond_input_shape[0],
        #                               self.sample_input_shape[1],
        #                                self.sample_input_shape[2],
        #                                self.sample_input_shape[3]])

        self.observational_mask = ObservationalMask(
            input_length=self.cond_input_shape[0]
        )

        # self.current_shape = self.current_shape.at[-1].set(self.current_shape[-1] + 1)

        self.first_projection = Projector(
            drop_rate=self.projection_dropout,
            output_channels=self.base_units,
        )

        # self.current_shape = self.current_shape.at[-1].set(self.base_units)

        self.positional_embedding = PositionalEmbedding(
            input_shape=(self.sample_input_shape[0] + self.cond_input_shape[0],
                         self.sample_input_shape[1],
                         self.sample_input_shape[2],
                         self.base_units),
        )

        self.init_time_embedding = TimeEmbeddingInit(
            time_embedding_channels=self.base_units * self.time_embedding_channels_multiplier
        )

        # input attention loops
        self.input_attention_blocks = [[
            TimeEmbedding(
                output_channels=self.base_units,
                dropout=self.attention_dropout,
            ),

            CuboidAttentionBlock(
                input_shape=(
                    self.sample_input_shape[0] + self.cond_input_shape[0],
                    self.sample_input_shape[1],
                    self.sample_input_shape[2],
                    self.base_units
                ),
                attention_heads=self.attention_heads,
                attention_pattern=self.block_attention_pattern,
                padding_type=self.padding_type,
                qkv_bias=False,
                attention_dropout=self.attention_dropout,
                ffn_dropout=self.ffn_dropout,
                ffn_activation=self.ffn_activation,
                gated_projection=self.gated_ffn,
                use_inter_attention_ffn=self.use_inter_attention_ffn,
                use_relative_position=self.use_relative_position,
                use_final_projection=self.use_attention_final_projection
            )
        ] for _ in range(self.io_depth[0])]

        # down-sample
        self.down_sampler = PatchMerge3D(
            input_channels=self.base_units,
            down_sample_factors=(1, self.spatial_downsample_factor,
                                 self.spatial_downsample_factor),
            output_channels=self.base_units * (self.spatial_downsample_factor ** 2)
        )

        # self.current_shape = self.current_shape.at[1].set(self.current_shape[1] / self.spatial_downsample_factor)
        # self.current_shape = self.current_shape.at[2].set(self.current_shape[2] / self.spatial_downsample_factor)
        # self.current_shape = self.current_shape.at[-1].set(
        #    self.current_shape[-1] * (self.spatial_downsample_factor ** 2))

        # bottleneck attention loops
        self.bottleneck_attention_blocks = [[
            TimeEmbedding(
                output_channels=self.base_units * (self.spatial_downsample_factor ** 2),
                dropout=self.attention_dropout,
            ),

            CuboidAttentionBlock(
                input_shape=(
                    self.sample_input_shape[0] + self.cond_input_shape[0],
                    self.sample_input_shape[1] / self.spatial_downsample_factor,
                    self.sample_input_shape[2] / self.spatial_downsample_factor,
                    self.base_units * (self.spatial_downsample_factor ** 2)
                ),
                attention_heads=self.attention_heads,
                attention_pattern=self.block_attention_pattern,
                padding_type=self.padding_type,
                qkv_bias=False,
                attention_dropout=self.attention_dropout,
                ffn_dropout=self.ffn_dropout,
                ffn_activation=self.ffn_activation,
                gated_projection=self.gated_ffn,
                use_inter_attention_ffn=self.use_inter_attention_ffn,
                use_relative_position=self.use_relative_position,
                use_final_projection=self.use_attention_final_projection
            )
        ] for _ in range(self.bottleneck_depth)]

        # up-sample
        self.up_sampler = UpSampler(
            output_channels=self.base_units,
            target_size=(self.sample_input_shape[0] + self.cond_input_shape[0],
                         self.sample_input_shape[1],
                         self.sample_input_shape[2])
        )

        # self.current_shape = self.current_shape.at[1].set(self.current_shape[1] * self.spatial_downsample_factor)
        # self.current_shape = self.current_shape.at[2].set(self.current_shape[2] * self.spatial_downsample_factor)
        # self.current_shape = self.current_shape.at[-1].set(
        #    self.current_shape[-1] / (self.spatial_downsample_factor ** 2))

        # output attention loop
        self.output_attention_blocks = [[
            TimeEmbedding(
                output_channels=self.base_units,
                dropout=self.attention_dropout,
            ),

            CuboidAttentionBlock(
                input_shape=(self.sample_input_shape[0] + self.cond_input_shape[0],
                             self.sample_input_shape[1],
                             self.sample_input_shape[2],
                             self.base_units),
                attention_heads=self.attention_heads,
                attention_pattern=self.block_attention_pattern,
                padding_type=self.padding_type,
                qkv_bias=False,
                attention_dropout=self.attention_dropout,
                ffn_dropout=self.ffn_dropout,
                ffn_activation=self.ffn_activation,
                gated_projection=self.gated_ffn,
                use_inter_attention_ffn=self.use_inter_attention_ffn,
                use_relative_position=self.use_relative_position,
                use_final_projection=self.use_attention_final_projection
            )
        ] for _ in range(self.io_depth[1])]

        self.final_projection = nn.Dense(self.cond_input_shape[-1])

        # self.current_shape = self.current_shape.at[-1].set(self.cond_input_shape[-1])

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:

        if ((x.shape[1:] != self.sample_input_shape) or (cond.shape[1:] != self.cond_input_shape)
                or (t.shape[0] != x.shape[0])):
            print(x.shape[1:], self.sample_input_shape)
            print(cond.shape[1:], self.cond_input_shape)
            print(t.shape, x.shape[0])
            raise ValueError

        x = jnp.concat([cond, x], axis=1)
        x = self.observational_mask(x)
        x = self.first_projection(x, train)
        x = self.positional_embedding(x)

        time_embedding = self.init_time_embedding(t)

        if self.unet_use_residual_connection:
            residual = x

        for block in self.input_attention_blocks:
            x = block[0](x, time_embedding, train)
            x = block[1](x, train)

        x = self.down_sampler(x)

        for block in self.bottleneck_attention_blocks:
            x = block[0](x, time_embedding, train)
            x = block[1](x, train)

        x = self.up_sampler(x)

        if self.unet_use_residual_connection:
            x += residual

        for block in self.output_attention_blocks:
            x = block[0](x, time_embedding, train)
            x = block[1](x, train)

        x = self.final_projection(x[:, self.cond_input_shape[0]:, ...])

        return x


#print(EarthformerUNet((5, 16, 16, 4), (5, 16, 16, 4), 256)
#      .tabulate(jax.random.PRNGKey(1), jnp.zeros((10, 5, 16, 16, 4)), jnp.zeros((10, 5, 16, 16, 4)), jnp.zeros(10),
#                False, depth=1, console_kwargs={'width': 150}))
