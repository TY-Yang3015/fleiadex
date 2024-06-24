# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VAE model definitions."""
import flax.linen as nn
import jax.numpy as jnp
from jax import random

class Encoder(nn.Module):
    latent_dim: int

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


class Decoder(nn.Module):
    original_shape: tuple = (128, 128, 3)

    def setup(self):
        self.fc1 = nn.Dense(16 * 16 * 128, name='fc1')
        self.deconv1 = nn.ConvTranspose(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                                        name='deconv1', kernel_init=nn.initializers.glorot_normal())
        self.deconv2 = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='deconv2',
                                        kernel_init=nn.initializers.glorot_normal())
        self.deconv3 = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='deconv3',
                                        kernel_init=nn.initializers.glorot_normal())
        self.deconv4 = nn.ConvTranspose(features=self.original_shape[2], kernel_size=(3, 3), strides=(1, 1),
                                        padding='SAME',
                                        name='deconv4', kernel_init=nn.initializers.glorot_normal())

    def __call__(self, x):
        # print("Latent shape before processing:", x.shape)
        x = self.fc1(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], 16, 16, 128))  # Reshape to match the expected input shape of deconv layers
        x = self.deconv1(x)
        x = nn.relu(x)
        x = self.deconv2(x)
        x = nn.relu(x)
        x = self.deconv3(x)
        x = nn.relu(x)
        x = self.deconv4(x)
        x = nn.sigmoid(x) * 2. - 1.  # Use sigmoid to map the output to the range [0, 1]
        # print("Output shape after decoding:", x.shape)
        return x


class VAE(nn.Module):
    """Full VAE model."""

    latents: int

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return self.decoder(z)


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


def model(latents):
    return VAE(latents=latents)