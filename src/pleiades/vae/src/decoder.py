import flax.linen as nn
import jax.numpy as jnp
import jax

from src.pleiades.blocks import (SelfAttention, ResNetBlock, Identity)


class Decoder(nn.Module):
    latent_channels: int
    spatial_upsample_schedule: tuple[int] = (2, 2, 2)
    channel_schedule: tuple[int] = (512, 256, 128)
    resnet_depth_schedule: tuple[int] = (3, 3, 3)
    attention_heads: int = 4
    attention_use_qkv_bias: bool = False
    attention_use_dropout: bool = True
    attention_dropout_rate: float = 0.1
    pre_output_resnet_depth: int = 3
    reconstruction_channels: int = 4
    conv_kernel_sizes: tuple[int] = (3, 3)

    def setup(self) -> None:

        if not (len(self.channel_schedule) == len(self.spatial_upsample_schedule)
                == len(self.resnet_depth_schedule)):
            raise ValueError('number fo spatial upsamplers must be equal to the number of resnet blocks'
                             'and channel schedule length.')

        self.conv_projection = nn.Sequential([
            nn.Conv(features=self.latent_channels,
                    kernel_size=self.conv_kernel_sizes,
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal()),
            nn.Conv(features=self.channel_schedule[0],
                    kernel_size=self.conv_kernel_sizes,
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal())
        ])

        self.attention = SelfAttention(
            output_channels=self.channel_schedule[0],
            attention_heads=self.attention_heads,
            use_qkv_bias=self.attention_use_qkv_bias,
            use_dropout=self.attention_use_dropout,
            dropout_rate=self.attention_dropout_rate,
        )

        resnet_block_lists = []
        upsampler_lists = []
        for i in range(len(self.spatial_upsample_schedule)):
            res_blocks = []
            res_blocks.append(ResNetBlock(
                output_channels=self.channel_schedule[i],
            ))
            for _ in range(self.resnet_depth_schedule[i] - 1):
                res_blocks.append(ResNetBlock())
            resnet_block_lists.append(res_blocks)
            upsampler_lists.append(
                nn.ConvTranspose(features=self.channel_schedule[i],
                                 kernel_size=self.conv_kernel_sizes,
                                 strides=(self.spatial_upsample_schedule[i],
                                          self.spatial_upsample_schedule[i]),
                                 padding='SAME',
                                 kernel_init=nn.initializers.kaiming_normal())
            )
        #upsampler_lists.append(Identity())

        #res_blocks = []
        #res_blocks.append(ResNetBlock(
        #    output_channels=self.channel_schedule[-1]
        #))
        #for _ in range(self.resnet_depth_schedule[-1] - 1):
        #    res_blocks.append(ResNetBlock())
        #resnet_block_lists.append(res_blocks)

        self.resnet_block_lists = resnet_block_lists
        self.upsampler_lists = upsampler_lists

        final_res_blocks = []
        for _ in range(self.pre_output_resnet_depth):
            final_res_blocks.append(ResNetBlock())

        self.final_res_blocks = final_res_blocks

        self.output_gr = nn.GroupNorm(num_groups=32 if self.channel_schedule[-1] % 32 == 0 \
            else self.channel_schedule[-1],
                                      group_size=None)

        self.output_conv = nn.Conv(features=self.reconstruction_channels,
                                   kernel_size=self.conv_kernel_sizes,
                                   strides=(1, 1),
                                   padding='SAME',
                                   kernel_init=nn.initializers.kaiming_normal()
                                   )

    @nn.compact
    def __call__(self, x, train:bool):

        if x.shape[-1] != self.latent_channels:
            raise ValueError()

        x = self.conv_projection(x)
        x = self.attention(x, train)

        for res_blocks, upsampler in zip(self.resnet_block_lists, self.upsampler_lists):
            for res_block in res_blocks:
                x = res_block(x)
            x = upsampler(x)

        for block in self.final_res_blocks:
            x = block(x)

        x = self.output_gr(x)
        x = nn.silu(x)
        x = self.output_conv(x)
        return x


# print(Decoder(3).tabulate(jax.random.PRNGKey(0), jnp.zeros((10, 16, 16, 3)), False,
#                          depth=1, console_kwargs={'width': 150}))
