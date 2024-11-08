import flax.linen as nn
import jax.numpy as jnp

from fleiadex.blocks import (
    TimeEmbedding2D,
    TimeEmbeddingInit,
    TimeEmbeddingResBlock,
    SelfAttention,
    Identity,
    PatchMerge3D,
    UpSampler2D,
)


class UNet4DConv2D(nn.Module):
    sample_input_shape: tuple[int, int, int, int] = (32, 32, 1)
    cond_input_shape: tuple[int, int, int, int] = (32, 32, 3)
    spatial_down_sample_schedule: tuple[int] = (2, 2)
    unet_use_residual: bool = True
    down_sampler_type: str = "conv"
    down_sample_resnet_depth: int = 2
    down_sample_use_attention: bool = True
    down_sample_attention_heads: int = 4
    down_sample_attention_use_memory_efficient: bool = True
    down_sample_attention_use_dropout: bool = True
    down_sample_attention_dropout_rate: float = 0.1
    bottleneck_resnet_depth: int = 2
    bottleneck_use_attention: bool = True
    bottleneck_attention_heads: int = 4
    bottleneck_attention_use_memory_efficient: bool = True
    bottleneck_attention_use_dropout: bool = True
    bottleneck_attention_dropout_rate: float = 0.1
    up_sampler_type: str = "interp2d"
    up_sample_resnet_depth: int = 2
    up_sample_use_attention: bool = True
    up_sample_attention_heads: int = 4
    up_sample_attention_use_memory_efficient: bool = True
    up_sample_attention_use_dropout: bool = True
    up_sample_attention_dropout_rate: float = 0.1

    dtype: jnp.dtype = jnp.float64

    def setup(self):
        self.base_channels = self.cond_input_shape[-1] + self.sample_input_shape[-1]

        assert ((self.sample_input_shape[0] == self.cond_input_shape[0])
                and (self.cond_input_shape[1] == self.sample_input_shape[1])), ("the sample and condition size"
                                                                                "must be the same.")

        self.input_conv_proj = nn.Conv(
            features=self.base_channels,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding="SAME",
        )

        self.time_embedding_init = TimeEmbeddingInit(
            time_embedding_channels=self.base_channels
        )
        self.time_embedding = TimeEmbedding2D(
            output_channels=self.base_channels, dropout=0
        )

        # down-sample
        down_sample_resnet_blocks = []
        down_sample_attention_block = []
        down_samplers = []
        current_channels = self.base_channels
        for _, down_sample_factor in enumerate(self.spatial_down_sample_schedule):
            res_block = []
            for _ in range(self.down_sample_resnet_depth):
                res_block.append(
                    TimeEmbeddingResBlock(output_channels=current_channels, dropout=0.1)
                )
            down_sample_resnet_blocks.append(res_block)

            if self.down_sample_use_attention:
                down_sample_attention_block.append(
                    SelfAttention(
                        output_channels=current_channels,
                        num_heads=self.down_sample_attention_heads,
                        use_memory_efficient_attention=self.down_sample_attention_use_memory_efficient,
                        use_dropout=self.down_sample_attention_use_dropout,
                        dropout_rate=self.down_sample_attention_dropout_rate,
                    )
                )
            else:
                down_sample_attention_block.append(Identity())

            if self.down_sampler_type == "conv":
                current_channels *= down_sample_factor ** 2
                down_samplers.append(
                    nn.Conv(
                        features=current_channels,
                        kernel_size=(3, 3),
                        strides=(down_sample_factor, down_sample_factor),
                        padding="SAME",
                    )
                )
            elif self.down_sampler_type == "patch_merge":
                down_samplers.append(
                    PatchMerge3D(
                        input_channels=current_channels,
                        output_channels=current_channels * down_sample_factor ** 2,
                        down_sample_factors=(1, down_sample_factor, down_sample_factor),
                    )
                )
                current_channels *= down_sample_factor ** 2
            else:
                raise NotImplementedError(
                    'only "conv" and "patch_merge" are supported for down-sampling.'
                )

        self.down_sample_resnet_blocks = down_sample_resnet_blocks
        self.down_sample_attention_block = down_sample_attention_block
        self.down_samplers = down_samplers

        # bottleneck
        bottleneck_resnet_blocks = []
        bottleneck_attention_block = []
        for _ in range(self.bottleneck_resnet_depth):
            bottleneck_resnet_blocks.append(
                TimeEmbeddingResBlock(output_channels=current_channels, dropout=0.1)
            )
        if self.bottleneck_use_attention:
            bottleneck_attention_block.append(
                SelfAttention(
                    output_channels=current_channels,
                    num_heads=self.bottleneck_attention_heads,
                    use_memory_efficient_attention=self.bottleneck_attention_use_memory_efficient,
                    use_dropout=self.bottleneck_attention_use_dropout,
                    dropout_rate=self.bottleneck_attention_dropout_rate,
                )
            )
        else:
            bottleneck_attention_block.append(Identity())
        for _ in range(self.bottleneck_resnet_depth):
            bottleneck_resnet_blocks.append(
                TimeEmbeddingResBlock(output_channels=current_channels, dropout=0.1)
            )

        self.bottleneck_resnet_blocks = bottleneck_resnet_blocks
        self.bottleneck_attention_block = bottleneck_attention_block

        # calculate encoded size
        latent_size = self.cond_input_shape[1]
        assert self.cond_input_shape[1] == self.cond_input_shape[0]
        for down_sample_factor in self.spatial_down_sample_schedule:
            latent_size /= down_sample_factor
        latent_size = int(latent_size)

        # up-sample
        up_sample_resnet_blocks = []
        up_sample_attention_block = []
        up_samplers = []
        for _, down_sample_factor in enumerate(
                reversed(self.spatial_down_sample_schedule)
        ):
            res_block = []
            for _ in range(self.up_sample_resnet_depth):
                res_block.append(
                    TimeEmbeddingResBlock(output_channels=current_channels, dropout=0.1)
                )
            up_sample_resnet_blocks.append(res_block)

            if self.up_sample_use_attention:
                up_sample_attention_block.append(
                    SelfAttention(
                        output_channels=current_channels,
                        num_heads=self.up_sample_attention_heads,
                        use_memory_efficient_attention=self.up_sample_attention_use_memory_efficient,
                        use_dropout=self.up_sample_attention_use_dropout,
                        dropout_rate=self.up_sample_attention_dropout_rate,
                    )
                )
            else:
                up_sample_attention_block.append(Identity())

            if self.up_sampler_type == "conv_transpose":
                current_channels /= down_sample_factor ** 2
                current_channels = int(current_channels)
                up_samplers.append(
                    nn.ConvTranspose(
                        features=current_channels,
                        kernel_size=(3, 3),
                        strides=(down_sample_factor, down_sample_factor),
                        padding="SAME",
                    )
                )
            elif self.up_sampler_type == "interp2d":
                up_samplers.append(
                    UpSampler2D(
                        output_channels=int(
                            current_channels / down_sample_factor ** 2
                        ),
                        target_size=(
                            latent_size * down_sample_factor,
                            latent_size * down_sample_factor,
                        ),
                    )
                )
                latent_size *= down_sample_factor
                latent_size = int(latent_size)
                current_channels /= down_sample_factor ** 2
                current_channels = int(current_channels)
            else:
                raise NotImplementedError(
                    'only "conv_transpose" and "interp2d" are supported for up-sampling.'
                )

        self.up_sample_resnet_blocks = up_sample_resnet_blocks
        self.up_sample_attention_block = up_sample_attention_block
        self.up_samplers = up_samplers

        self.final_projection = nn.Dense(features=self.sample_input_shape[-1])

    def __call__(
            self, x: jnp.ndarray, cond: jnp.ndarray, t: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        x = jnp.concat([cond, x], axis=-1)
        x = self.input_conv_proj(x)

        time_embedding = self.time_embedding_init(t)
        x = self.time_embedding(x, time_embedding, train)

        if self.unet_use_residual:
            residual = []
        for i in range(len(self.spatial_down_sample_schedule)):
            for res_block in self.down_sample_resnet_blocks[i]:
                x = res_block(x, time_embedding, train)
                x = self.down_sample_attention_block[i](x, train)
            x = self.down_samplers[i](x)
            if self.unet_use_residual:
                residual.append(x)

        # bottleneck
        for i in range(self.bottleneck_resnet_depth):
            x = self.bottleneck_resnet_blocks[i](x, time_embedding, train)

        x = self.bottleneck_attention_block[0](x, train)

        for i in range(self.bottleneck_resnet_depth):
            x = self.bottleneck_resnet_blocks[-i - 1](x, time_embedding, train)

        # up-sample phase
        for i in range(len(self.spatial_down_sample_schedule)):
            if self.unet_use_residual:
                x = jnp.concat([residual[-i - 1], x], axis=-1)
            for res_block in self.up_sample_resnet_blocks[i]:
                x = res_block(x, time_embedding, train)
                x = self.up_sample_attention_block[i](x, train)
            x = self.up_samplers[i](x)

        x = self.final_projection(x[..., self.cond_input_shape[-1]:])

        return x

#import jax
#print(
#     UNet4DConv2D().tabulate(
#         jax.random.PRNGKey(1),
#        jnp.zeros((8, 32, 32, 1)),
#         jnp.zeros((8, 32, 32, 3)),
#         jnp.zeros(8),
#         False,
#         depth=1,
#         console_kwargs={'width': 150}
#     )
#)
