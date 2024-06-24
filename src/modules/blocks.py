import tensorflow as tf
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = tf.math.log(10000.) / (half_dim - 1)
    emb = tf.math.exp(np.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1)
    return emb


class DiffusionNN(tf.keras.Model):
    def __init__(self):
        super(DiffusionNN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(14, activation='relu')
        self.conv1 = tf.keras.layers.Conv2D(14, (3, 3), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(7, (3, 3), padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(1, (3, 3), padding='same')

    def call(self, inputs, training=False):
        x, t_emb = inputs

        t_emb = self.dense1(t_emb)
        t_emb = tf.reshape(t_emb, [-1, 1, 1, t_emb.shape[-1]])
        t_emb = tf.tile(t_emb, [1, tf.shape(x)[1], tf.shape(x)[2], 1])

        x = self.conv1(x)
        x = self.conv2(x + t_emb)  # Add timestep embedding
        x = self.conv4(x)
        return x
