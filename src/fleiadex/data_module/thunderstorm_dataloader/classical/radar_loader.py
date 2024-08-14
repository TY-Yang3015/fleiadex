import tensorflow as tf
import jax.numpy as jnp
from glob import glob
import matplotlib.pyplot as plt


class RadarDataModule:
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
        dataset = jnp.where(dataset < 0.0, 0.0, dataset)
        dataset = jnp.where(dataset > 60.0, 60.0, dataset)
        return dataset

    def load_dataset(self, month_str, num_samples):
        data_dir = self.data_root + month_str
        data_paths = glob("./data/" + data_dir + ".npy")
        dataset = []
        for path in data_paths:
            dataset.append((jnp.load(path)[: num_samples * 2 : 2]))
        dataset = jnp.array(dataset)
        dataset = jnp.reshape(dataset, dataset.shape[1:])
        dataset_processed = self.preprocessing_fn(dataset)
        # for i in range(dataset.shape[0]):
        #     plt.imshow(dataset_processed[i])
        #     plt.colorbar()
        #     plt.show()
        return dataset_processed
