import flax.linen as nn
import jax.numpy as jnp
from jax import random
from omegaconf import DictConfig


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
        x = nn.sigmoid(x)
        x = (x * (self.config.data_spec.clip_max - self.config.data_spec.clip_min)
             + self.config.data_spec.clip_min)
        return x


@nn.jit
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
