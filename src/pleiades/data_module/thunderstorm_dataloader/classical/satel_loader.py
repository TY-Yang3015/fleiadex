import tensorflow as tf
import jax.numpy as jnp
from glob import glob
import os


class SatelDataModule:
    def __init__(
        self,
        data_root,
        input_image_size=(128, 128),
        output_image_size=(128, 128),
    ):
        self.data_root = data_root
        self.input_image_size = tf.constant(input_image_size)
        self.output_image_size = tf.constant(output_image_size)

    def preprocessing_fn(self, dataset):
        dataset = jnp.mean(dataset, axis=1)
        return dataset

    def load_dataset(self, month_str, num_samples):
        data_dir = self.data_root + month_str
        data_paths = glob("./data/" + data_dir + "bandopt00.npy")
        dataset = []
        for path in data_paths:
            dataset.append((jnp.load(path)[:num_samples]))
        dataset = jnp.array(dataset)
        dataset = jnp.reshape(dataset, dataset.shape[1:])
        dataset = self.preprocessing_fn(dataset)
        return dataset
