import flax.linen as nn
import jax.numpy as jnp
from jax import random

from omegaconf import DictConfig


class LegacyEncoder(nn.Module):
    config: DictConfig

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2),
                             name='conv1', kernel_init=nn.initializers.glorot_normal())
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), name='conv2',
                             kernel_init=nn.initializers.glorot_normal())
        self.conv3 = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), name='conv3',
                             kernel_init=nn.initializers.glorot_normal())
        self.conv4 = nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2), name='conv4',
                             kernel_init=nn.initializers.glorot_normal())
        self.fc_mean = nn.Dense(self.latent_dim, name='fc_mean')
        self.fc_logvar = nn.Dense(self.latent_dim, name='fc_logvar')

    def __call__(self, x):
        # print("Input shape before processing:", x.shape)
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = self.conv3(x)
        x = nn.relu(x)
        x = self.conv4(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        # print("Shape after flattening:", x.shape)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        # print("Mean shape:", mean.shape, "Logvar shape:", logvar.shape)
        return mean, logvar


class Encoder(nn.Module):
    config: DictConfig

    def setup(self):
        n_layers = list([])
        for i in range(self.config.nn_spec.num_of_layers):
            layer = nn.Conv(features=self.config.nn_spec.features[i],
                            kernel_size=(self.config.nn_spec.kernel_size,
                                         self.config.nn_spec.kernel_size),
                            strides=(self.config.nn_spec.stride,
                                     self.config.nn_spec.stride),
                            name=f'conv{i + 1}',
                            kernel_init=nn.initializers.glorot_normal())
            n_layers.append(layer)
        self.n_layers = n_layers
        self.fc_mean = nn.Dense(self.config.nn_spec.latents, name='fc_mean')
        self.fc_logvar = nn.Dense(self.config.nn_spec.latents, name='fc_logvar')

    def __call__(self, x):
        for layer in self.n_layers:
            x = layer(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    config: DictConfig

    def setup(self):
        self.n = self.config.data_spec.image_size / (self.config.nn_spec.stride ** self.config.nn_spec.num_of_layers)
        self.feature_max = self.config.nn_spec.max_feature

        self.fc1 = nn.Dense(int(self.config.nn_spec.decoder_input), name='fc1')
        n_layers = []
        for i in reversed(range(self.config.nn_spec.num_of_layers)):
            layer = nn.ConvTranspose(features=self.config.nn_spec.features[i],
                                     kernel_size=(self.config.nn_spec.kernel_size,
                                                  self.config.nn_spec.kernel_size),
                                     strides=(self.config.nn_spec.stride,
                                              self.config.nn_spec.stride),
                                     padding='SAME',
                                     name=f'deconv_{int(self.config.nn_spec.num_of_layers - i)}',
                                     kernel_init=nn.initializers.glorot_normal())
            n_layers.append(layer)
        self.n_layers = n_layers
        self.deconv_final = nn.ConvTranspose(features=self.config.data_spec.image_channels,
                                             kernel_size=(self.config.nn_spec.kernel_size,
                                                          self.config.nn_spec.kernel_size),
                                             strides=(1, 1),
                                             padding='SAME',
                                             name='deconv_final',
                                             kernel_init=nn.initializers.glorot_normal())

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], int(self.n), int(self.n), self.feature_max))
        for layer in self.n_layers:
            x = layer(x)
            x = nn.relu(x)
        x = self.deconv_final(x)
        x = nn.sigmoid(x) * 2. - 1.
        return x


class LegacyDecoder(nn.Module):
    config: DictConfig

    def setup(self):
        self.fc1 = nn.Dense(8 * 8 * 256, name='fc1')
        self.deconv1 = nn.ConvTranspose(features=256, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                                        name='deconv1', kernel_init=nn.initializers.glorot_normal())
        self.deconv2 = nn.ConvTranspose(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                                        name='deconv2',
                                        kernel_init=nn.initializers.glorot_normal())
        self.deconv3 = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='deconv3',
                                        kernel_init=nn.initializers.glorot_normal())
        self.deconv4 = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='deconv4',
                                        kernel_init=nn.initializers.glorot_normal())
        self.deconvf = nn.ConvTranspose(features=self.config.data_spec.image_channels, kernel_size=(3, 3), strides=(1, 1),
                                        padding='SAME',
                                        name='deconvf', kernel_init=nn.initializers.glorot_normal())

    def __call__(self, x):
        # print("Latent shape before processing:", x.shape)
        x = self.fc1(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], 8, 8, 256))  # Reshape to match the expected input shape of deconv layers
        x = self.deconv1(x)
        x = nn.relu(x)
        x = self.deconv2(x)
        x = nn.relu(x)
        x = self.deconv3(x)
        x = nn.relu(x)
        x = self.deconv4(x)
        x = nn.relu(x)
        x = self.deconvf(x)
        x = nn.sigmoid(x) * 2. - 1.  # Use sigmoid to map the output to the range [0, 1]
        # print("Output shape after decoding:", x.shape)
        return x


class VAE(nn.Module):
    config: DictConfig

    def setup(self):
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = self.reparameterise(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return self.decoder(z)

    def reparameterise(self, rng, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, logvar.shape)
        return mean + eps * std


def model(config: DictConfig):
    return VAE(config)
