import flax.linen as nn
import jax.numpy as jnp
import jax
from fleiadex.utils import get_activation

from fleiadex.blocks import SelfAttention, ResNetBlock


class Decoder(nn.Module):
    latent_channels: int
    spatial_upsample_schedule: tuple[int] = (2, 2, 2)
    channel_schedule: tuple[int] = (512, 256, 128)
    resnet_depth_schedule: tuple[int] = (3, 3, 3)
    use_attention: bool = True
    attention_heads: int = 8
    attention_use_qkv_bias: bool = False
    attention_use_dropout: bool = True
    attention_dropout_rate: float = 0.1
    up_sampler_type: str = "resize"
    use_memory_efficient_attention: bool = True
    pre_output_resnet_depth: int = 3
    use_final_linear_projection: bool = False
    reconstruction_channels: int = 4
    conv_kernel_sizes: tuple[int] = (3, 3)
    up_sample_activation: str = "silu"
    pre_output_activation: str = "silu"
    final_activation: str = "silu"

    def setup(self) -> None:

        if not (
            len(self.channel_schedule)
            == len(self.spatial_upsample_schedule)
            == len(self.resnet_depth_schedule)
        ):
            raise ValueError(
                "number fo spatial upsamplers must be equal to the number of resnet blocks"
                "and channel schedule length."
            )

        self.conv_projection = nn.Sequential(
            [
                nn.Conv(
                    features=self.latent_channels,
                    kernel_size=self.conv_kernel_sizes,
                    strides=(1, 1),
                    padding="SAME",
                    kernel_init=nn.initializers.kaiming_normal(),
                ),
                nn.Conv(
                    features=self.channel_schedule[0],
                    kernel_size=self.conv_kernel_sizes,
                    strides=(1, 1),
                    padding="SAME",
                    kernel_init=nn.initializers.kaiming_normal(),
                ),
            ]
        )

        self.attention = SelfAttention(
            output_channels=self.channel_schedule[0],
            num_heads=self.attention_heads,
            use_qkv_bias=self.attention_use_qkv_bias,
            use_dropout=self.attention_use_dropout,
            dropout_rate=self.attention_dropout_rate,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
        )

        resnet_block_lists = []
        upsampler_lists = []
        for i in range(len(self.spatial_upsample_schedule)):
            res_blocks = []
            res_blocks.append(
                ResNetBlock(
                    output_channels=self.channel_schedule[i],
                    activation=self.up_sample_activation,
                )
            )
            for _ in range(self.resnet_depth_schedule[i] - 1):
                res_blocks.append(
                    ResNetBlock(
                        output_channels=self.channel_schedule[i],
                        activation=self.up_sample_activation,
                    )
                )
            resnet_block_lists.append(res_blocks)
            if self.up_sampler_type == "conv_trans":
                upsampler_lists.append(
                    nn.ConvTranspose(
                        features=self.channel_schedule[i],
                        kernel_size=self.conv_kernel_sizes,
                        strides=(
                            self.spatial_upsample_schedule[i],
                            self.spatial_upsample_schedule[i],
                        ),
                        padding="SAME",
                        kernel_init=nn.initializers.kaiming_normal(),
                    )
                )
            elif self.up_sampler_type == "resize":
                upsampler_lists.append(
                    nn.ConvTranspose(
                        features=self.channel_schedule[i],
                        kernel_size=self.conv_kernel_sizes,
                        strides=(1, 1),
                        padding="SAME",
                        kernel_init=nn.initializers.kaiming_normal(),
                    )
                )
            else:
                raise NotImplementedError(
                    'only "conv_trans" and "resize" are supported up-sample options.'
                )

        self.resnet_block_lists = resnet_block_lists
        self.upsampler_lists = upsampler_lists

        final_res_blocks = []
        for _ in range(self.pre_output_resnet_depth):
            final_res_blocks.append(
                ResNetBlock(
                    activation=self.pre_output_activation,
                )
            )

        self.final_res_blocks = final_res_blocks

        self.output_gr = nn.GroupNorm(
            num_groups=32
            if self.channel_schedule[-1] % 32 == 0
            else self.channel_schedule[-1],
            group_size=None,
        )

        self.output_conv = nn.Conv(
            features=self.reconstruction_channels,
            kernel_size=self.conv_kernel_sizes,
            strides=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.kaiming_normal(),
        )

        if self.use_final_linear_projection:
            self.final_linear_projection = nn.Dense(
                features=self.reconstruction_channels,
            )

    @nn.compact
    def __call__(self, x, train: bool):

        if x.shape[-1] != self.latent_channels:
            raise ValueError()

        x = self.conv_projection(x)

        if self.use_attention:
            x = self.attention(x, train)

        i = 0
        for res_blocks, upsampler in zip(self.resnet_block_lists, self.upsampler_lists):
            for res_block in res_blocks:
                x = res_block(x)
            if self.up_sampler_type == "resize":
                x = jax.image.resize(
                    x,
                    shape=(
                        x.shape[0],
                        x.shape[1] * self.spatial_upsample_schedule[i],
                        x.shape[2] * self.spatial_upsample_schedule[i],
                        self.channel_schedule[i],
                    ),
                    method="nearest",
                )
            x = upsampler(x)

        for block in self.final_res_blocks:
            x = block(x)

        x = self.output_gr(x)
        x = get_activation(self.final_activation)(x)
        x = self.output_conv(x)

        if self.use_final_linear_projection:
            x = self.final_linear_projection(x)

        return x


# print(Decoder(3).tabulate(jax.random.PRNGKey(0), jnp.zeros((10, 16, 16, 3)), False,
#                      depth=1, console_kwargs={'width': 150}))
