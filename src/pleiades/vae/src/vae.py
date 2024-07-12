import flax.core
import flax.linen as nn
import jax.numpy as jnp
import jax

from src.pleiades.vae.src.decoder import Decoder
from src.pleiades.vae.src.encoder import Encoder


class VAE(nn.Module):
    encoder_spatial_downsample_schedule: tuple[int] = (2, 2)
    encoder_channel_schedule: tuple[int] = (128, 256, 512)
    encoder_resnet_depth_schedule: tuple[int] = (2, 2, 2)
    encoder_attention_heads: int = 4
    encoder_attention_use_qkv_bias: bool = False
    encoder_attention_use_dropout: bool = True
    encoder_attention_dropout_rate: float = 0.1
    encoder_post_attention_resnet_depth: int = 2
    encoder_latents_channels: int = 3
    encoder_conv_kernel_sizes: tuple[int] = (3, 3)

    decoder_latent_channels: int = 3
    decoder_spatial_upsample_schedule: tuple[int] = (2, 2)
    decoder_channel_schedule: tuple[int] = (512, 256, 128)
    decoder_resnet_depth_schedule: tuple[int] = (2, 2, 2)
    decoder_attention_heads: int = 4
    decoder_attention_use_qkv_bias: bool = False
    decoder_attention_use_dropout: bool = True
    decoder_attention_dropout_rate: float = 0.1
    decoder_pre_output_resnet_depth: int = 3
    decoder_reconstruction_channels: int = 1
    decoder_conv_kernel_sizes: tuple[int] = (3, 3)

    def setup(self):
        self.encoder = Encoder(
            spatial_downsample_schedule=self.encoder_spatial_downsample_schedule,
            channel_schedule=self.encoder_channel_schedule,
            resnet_depth_schedule=self.encoder_resnet_depth_schedule,
            attention_heads=self.encoder_attention_heads,
            attention_use_qkv_bias=self.encoder_attention_use_qkv_bias,
            attention_use_dropout=self.encoder_attention_use_dropout,
            attention_dropout_rate=self.encoder_attention_dropout_rate,
            post_attention_resnet_depth=self.encoder_post_attention_resnet_depth,
            latents_channels=self.encoder_latents_channels,
            conv_kernel_sizes=self.encoder_conv_kernel_sizes
        )
        self.decoder = Decoder(
            latent_channels=self.decoder_latent_channels,
            spatial_upsample_schedule=self.decoder_spatial_upsample_schedule,
            channel_schedule=self.decoder_channel_schedule,
            resnet_depth_schedule=self.decoder_resnet_depth_schedule,
            attention_heads=self.decoder_attention_heads,
            attention_use_qkv_bias=self.decoder_attention_use_qkv_bias,
            attention_use_dropout=self.decoder_attention_use_dropout,
            attention_dropout_rate=self.decoder_attention_dropout_rate,
            pre_output_resnet_depth=self.decoder_pre_output_resnet_depth,
            reconstruction_channels=self.decoder_reconstruction_channels,
            conv_kernel_sizes=self.decoder_conv_kernel_sizes
        )

    def __call__(self, x, z_rng, train: bool):
        mean, logvar = self.encoder(x, train)
        z = self.reparameterise(z_rng, mean, logvar)
        recon_x = self.decoder(z, train)
        return recon_x, mean, logvar

    def generate(self, z):
        return self.decoder(z, False)

    def reparameterise(self, rng, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, logvar.shape)
        return mean + eps * std


# print(VAE().tabulate(jax.random.PRNGKey(0), jnp.ones((10, 128, 128, 4)),
#                     jax.random.PRNGKey(1), False))


def get_vae_instance(config: flax.core.FrozenDict):
    return VAE(**config)
