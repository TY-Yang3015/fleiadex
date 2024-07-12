import flax.linen as nn
import jax.numpy as jnp
import jax

from src.pleiades.blocks import (SelfAttention, ResNetBlock, Identity)


class Encoder(nn.Module):
    spatial_downsample_schedule: tuple[int] = (2, 2)
    channel_schedule: tuple[int] = (128, 256, 512)
    resnet_depth_schedule: tuple[int] = (2, 2, 2)
    attention_heads: int = 4
    attention_use_qkv_bias: bool = False
    attention_use_dropout: bool = True
    attention_dropout_rate: float = 0.1
    post_attention_resnet_depth: int = 2
    latents_channels: int = 3
    conv_kernel_sizes: tuple[int] = (3, 3)

    def setup(self) -> None:

        if (len(self.channel_schedule) - len(self.spatial_downsample_schedule) != 1
                or len(self.channel_schedule) != len(self.resnet_depth_schedule)):
            raise ValueError()

        self.conv_projection = nn.Conv(features=self.channel_schedule[0],
                                       kernel_size=self.conv_kernel_sizes,
                                       strides=(1, 1),
                                       padding='SAME',
                                       kernel_init=nn.initializers.kaiming_normal())

        resnet_block_lists = []
        downsampler_lists = []
        for i in range(len(self.resnet_depth_schedule) - 1):
            res_blocks = []
            res_blocks.append(ResNetBlock(
                output_channels=self.channel_schedule[i],
            ))
            for _ in range(self.resnet_depth_schedule[i] - 1):
                res_blocks.append(ResNetBlock())
            resnet_block_lists.append(res_blocks)
            downsampler_lists.append(
                nn.Conv(features=self.channel_schedule[i],
                        kernel_size=self.conv_kernel_sizes,
                        strides=(self.spatial_downsample_schedule[i],
                                 self.spatial_downsample_schedule[i]),
                        padding='SAME',
                        kernel_init=nn.initializers.kaiming_normal())
            )
        downsampler_lists.append(Identity())

        res_blocks = []
        res_blocks.append(ResNetBlock(
            output_channels=self.channel_schedule[-1]
        ))
        for _ in range(self.resnet_depth_schedule[-1] - 1):
            res_blocks.append(ResNetBlock())
        resnet_block_lists.append(res_blocks)

        self.attention = SelfAttention(
            output_channels=self.channel_schedule[-1],
            attention_heads=self.attention_heads,
            use_qkv_bias=self.attention_use_qkv_bias,
            use_dropout=self.attention_use_dropout,
            dropout_rate=self.attention_dropout_rate,
        )

        self.resnet_block_lists = resnet_block_lists
        self.downsampler_lists = downsampler_lists

        final_res_blocks = []
        for _ in range(self.post_attention_resnet_depth):
            final_res_blocks.append(ResNetBlock())

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

        x = self.attention(x, train)
        for block in self.final_res_blocks:
            x = block(x)

        x = self.output_gr(x)
        x = nn.silu(x)
        x = self.output_conv(x)

        mean, logvar = x[..., :3], x[..., 3:]

        return mean, logvar

#print(Encoder().tabulate(jax.random.PRNGKey(0), jnp.zeros((10, 64, 64, 1)), False,
#                         depth=1, console_kwargs={'width': 150}))
