import flax.core
import flax.linen as nn
import jax.numpy as jnp
import jax

from src.pleiades.vae.prediff_vae.decoder import Decoder
from src.pleiades.vae.prediff_vae.encoder import Encoder


class VAE(nn.Module):
    """
    the implementation of variational autoencoder with residual connections
    and self-attention blocks.

    for the usage of parameters, please refer to ``pleiades.vae.encoder``
    and ``pleiades.vae.decoder``.
    """
    encoder_spatial_downsample_schedule: tuple[int] = (2, 2, 2)
    encoder_channel_schedule: tuple[int] = (128, 256, 512)
    encoder_resnet_depth_schedule: tuple[int] = (2, 2, 2, 2)
    encoder_attention_heads: int = 8
    encoder_attention_use_qkv_bias: bool = False
    encoder_attention_use_dropout: bool = True
    encoder_attention_dropout_rate: float = 0.1
    encoder_use_memory_efficient_attention: bool = False
    encoder_post_attention_resnet_depth: int = 2
    encoder_latents_channels: int = 4
    encoder_conv_kernel_sizes: tuple[int] = (3, 3)
    encoder_down_sample_activation: str = 'silu'
    encoder_post_attention_activation: str = 'silu'
    encoder_final_activation: str = 'silu'

    decoder_latent_channels: int = 4
    decoder_spatial_upsample_schedule: tuple[int] = (2, 2, 2)
    decoder_channel_schedule: tuple[int] = (512, 256, 128)
    decoder_resnet_depth_schedule: tuple[int] = (3, 3, 3)
    decoder_attention_heads: int = 8
    decoder_attention_use_qkv_bias: bool = False
    decoder_attention_use_dropout: bool = True
    decoder_attention_dropout_rate: float = 0.1
    decoder_up_sampler_type: str = 'resize'
    decoder_use_memory_efficient_attention: bool = False
    decoder_pre_output_resnet_depth: int = 3
    decoder_use_final_linear_projection: bool = False
    decoder_reconstruction_channels: int = 4
    decoder_conv_kernel_sizes: tuple[int] = (3, 3)
    decoder_up_sample_activation: str = "silu"
    decoder_pre_output_activation: str = "silu"
    decoder_final_activation: str = "silu"

    def setup(self) -> None:
        self.encoder = Encoder(
            spatial_downsample_schedule=self.encoder_spatial_downsample_schedule,
            channel_schedule=self.encoder_channel_schedule,
            resnet_depth_schedule=self.encoder_resnet_depth_schedule,
            attention_heads=self.encoder_attention_heads,
            attention_use_qkv_bias=self.encoder_attention_use_qkv_bias,
            attention_use_dropout=self.encoder_attention_use_dropout,
            attention_dropout_rate=self.encoder_attention_dropout_rate,
            use_memory_efficient_attention=self.encoder_use_memory_efficient_attention,
            post_attention_resnet_depth=self.encoder_post_attention_resnet_depth,
            latents_channels=self.encoder_latents_channels,
            conv_kernel_sizes=self.encoder_conv_kernel_sizes,
            down_sample_activation=self.encoder_down_sample_activation,
            post_attention_activation=self.encoder_post_attention_activation,
            final_activation=self.encoder_final_activation
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
            up_sampler_type=self.decoder_up_sampler_type,
            use_memory_efficient_attention=self.decoder_use_memory_efficient_attention,
            pre_output_resnet_depth=self.decoder_pre_output_resnet_depth,
            use_final_linear_projection=self.decoder_use_final_linear_projection,
            reconstruction_channels=self.decoder_reconstruction_channels,
            conv_kernel_sizes=self.decoder_conv_kernel_sizes,
            up_sample_activation=self.decoder_up_sample_activation,
            pre_output_activation=self.decoder_pre_output_activation,
            final_activation=self.decoder_final_activation
        )

    def __call__(self, x: jnp.ndarray, z_rng: jnp.ndarray
                 , train: bool) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        :param x: The input image to be encoded with shape ``(batch_size,
            width, height, channels)``.
        :param z_rng: random number generator used to encode latents.
        :param train: bool. whether to use training or inference mode.

        :return: a tuple of jnp.ndarray, consists
            of the reconstructed image and the latents (mean and logarithmic variance).
        """
        mean, logvar = self.encoder(x, train)
        z = self._reparameterise(z_rng, mean, logvar)
        recon_x = self.decoder(z, train)
        return recon_x, mean, logvar

    def generate(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        :param z: the latents to be used for image generation. make sure this
            matches the specified latents dimensions.

        :return: the reconstructed images.
        """
        return self.decoder(z, train=False)

    def _reparameterise(self, rng: jnp.ndarray, mean: jnp.ndarray,
                        logvar: jnp.ndarray) -> jnp.ndarray:
        """

        the reparameterisation trick. use the equation mean + (epsilon + log(0.5 * log_var)).

        :param rng: the random number generator used to generate epsilon noise.
        :param mean: encoded means.
        :param logvar: encoded logarithmic variances.

        :return: the reparameterised latents.
        """
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, logvar.shape)
        return mean + eps * std

    def encode(self, x: jnp.ndarray, z_rng: jax.random.PRNGKey) -> jnp.ndarray:
        mean, logvar = self.encoder(x, False)
        z = self._reparameterise(z_rng, mean, logvar)
        return z

    def decode(self, z):
        recon_x = self.decoder(z, False)
        return recon_x


# print(VAE().tabulate(jax.random.PRNGKey(0), jnp.ones((10, 128, 128, 4)),
#                  jax.random.PRNGKey(1), False, console_kwargs={'width': 300}, depth=1))