import flax.linen as nn
import jax.numpy as jnp
import jax

from src.pleiades.blocks import (SelfAttention, ResNetBlock)
from src.pleiades.utils import get_activation


class Encoder(nn.Module):
    """

    """

    spatial_downsample_schedule: tuple[int] = (2, 2, 2)
    channel_schedule: tuple[int] = (128, 256, 512)
    resnet_depth_schedule: tuple[int] = (2, 2, 2, 2)
    attention_heads: int = 8
    attention_use_qkv_bias: bool = False
    attention_use_dropout: bool = True
    attention_dropout_rate: float = 0.1
    use_memory_efficient_attention: bool = False
    post_attention_resnet_depth: int = 2
    latents_channels: int = 4
    conv_kernel_sizes: tuple[int] = (3, 3)
    down_sample_activation: str = 'silu'
    post_attention_activation: str = 'silu'
    final_activation: str = 'silu'


    def setup(self) -> None:

        if len(self.resnet_depth_schedule) - len(self.spatial_downsample_schedule) != 1:
            raise ValueError('resnet depth schedule must be longer than downsampler schedule length'
                             'by 1, of which is the depth of the resnet block after the last'
                             'downsampler.')

        if len(self.spatial_downsample_schedule) != len(self.channel_schedule):
            raise ValueError("channel schedule length and downsampler schedule length must be equal.")

        self.conv_projection = nn.Conv(features=self.channel_schedule[0],
                                       kernel_size=self.conv_kernel_sizes,
                                       strides=(1, 1),
                                       padding='SAME',
                                       kernel_init=nn.initializers.kaiming_normal())

        resnet_block_lists = []
        downsampler_lists = []
        for i in range(len(self.spatial_downsample_schedule)):
            res_blocks = []
            res_blocks.append(ResNetBlock(
                output_channels=self.channel_schedule[i],
                activation=self.down_sample_activation
            ))
            for _ in range(self.resnet_depth_schedule[i] - 1):
                res_blocks.append(ResNetBlock(
                    output_channels=self.channel_schedule[i],
                    activation=self.down_sample_activation
                ))
            resnet_block_lists.append(res_blocks)
            downsampler_lists.append(
                nn.Conv(features=self.channel_schedule[i],
                        kernel_size=self.conv_kernel_sizes,
                        strides=(self.spatial_downsample_schedule[i],
                                 self.spatial_downsample_schedule[i]),
                        padding='SAME',
                        kernel_init=nn.initializers.kaiming_normal())
            )

        res_blocks = []
        res_blocks.append(ResNetBlock(
            output_channels=self.channel_schedule[-1],
            activation=self.down_sample_activation
        ))
        for _ in range(self.resnet_depth_schedule[-1] - 1):
            res_blocks.append(ResNetBlock(
                output_channels=self.channel_schedule[-1],
                activation=self.down_sample_activation
            ))
        resnet_block_lists.append(res_blocks)

        self.attention = SelfAttention(
            output_channels=self.channel_schedule[-1],
            num_heads=self.attention_heads,
            use_qkv_bias=self.attention_use_qkv_bias,
            use_dropout=self.attention_use_dropout,
            dropout_rate=self.attention_dropout_rate,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
        )

        self.resnet_block_lists = resnet_block_lists
        self.downsampler_lists = downsampler_lists

        final_res_blocks = []
        for _ in range(self.post_attention_resnet_depth):
            final_res_blocks.append(ResNetBlock(
                activation=self.post_attention_activation
            ))

        self.final_res_blocks = final_res_blocks

        self.output_gr = nn.GroupNorm(num_groups=32 if self.channel_schedule[-1] % 32 == 0 \
            else self.channel_schedule[-1],
                                      group_size=None)

        self.output_conv = nn.Sequential([
            nn.Conv(features=self.latents_channels * 2,
                    kernel_size=self.conv_kernel_sizes,
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal()
                    ),
            nn.Conv(features=self.latents_channels * 2,
                    kernel_size=self.conv_kernel_sizes,
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal()
                    )])

    @nn.compact
    def __call__(self, x, train: bool):
        x = self.conv_projection(x)

        for res_blocks, downsampler in zip(self.resnet_block_lists, self.downsampler_lists):
            for res_block in res_blocks:
                x = res_block(x)
            x = downsampler(x)

        for res_block in self.resnet_block_lists[-1]:
            x = res_block(x)

        x = self.attention(x, train)
        for block in self.final_res_blocks:
            x = block(x)

        x = self.output_gr(x)
        x = get_activation(self.final_activation)(x)
        x = self.output_conv(x)

        mean, logvar = x[..., :self.latents_channels], x[..., self.latents_channels:]

        return mean, logvar


# print(Encoder().tabulate(jax.random.PRNGKey(0), jnp.zeros((10, 128, 128, 4)), False,
#                         depth=1, console_kwargs={'width': 150}))
