import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class DataLoader:
    def __init__(
        self,
        dataset_dir,
        batch_size=1,
        sequence_length=1,
        mean=[264.5569, 249.6994, 228.8998, 242.5480],
        std=[25.1830, 18.3450, 7.5322, 13.1226],
        input_image_size=(128, 128),
        output_image_size=(128, 128),
        image_channels=4,
        *args,
        **kwargs,
    ):
        """
        Datamodule for loading SATEL dataset.

        SATEL dataset have arbiturary C input channels. The mean and std are calculated for 4 channels.
        We need to return a size of (B, T, H, W, C) for each batch.

        To train the autoencoder, we will resize the dataset in the ```fit()``` function
        to (B x T, H, W, C). In this case, we view the temporal frames as the batch size.
        Notice that in this way, we are letting the autoencoder model learn without the temporal information.

        To train the diffusion network, we will resize the dataset in the ```fit()``` function
        to (B, H, W, T x C). In this case, we view the temporal frames as additional channels.
        Notice that in this way, we are letting the diffusion model learn with the temporal information.

        Args:
            dataset_dir (str): Path to the root directory of the dataset.
            batch_size (int): Batch size.
            sequence_length (int): Length of the sequence.
            mean (list): Mean value for normalization.
            std (list): Standard deviation value for normalization.
            input_image_size (tuple): Size of the input image.
            output_image_size (tuple): Size of the output image.

        """
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_image_size = tf.constant(input_image_size)
        self.output_image_size = tf.constant(output_image_size)

        self.image_channels = image_channels
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        assert (
            len(self.mean) == self.image_channels
        ), f"Mean should have {image_channels} channels"
        assert (
            len(self.std) == self.image_channels
        ), f"Std should have {image_channels} channels"

        self.train_dataset = self.read_catalog(
            os.path.join(dataset_dir, "CATALOG.csv"), splits="train"
        )
        self.val_dataset = self.read_catalog(
            os.path.join(dataset_dir, "CATALOG.csv"), splits="val"
        )

    def read_catalog(self, catalog_path, splits="train"):
        """
        Read the catalog file and return the list of image paths.

        Args:
            catalog_path (str): Path to the catalog file.
        """
        df = pd.read_csv(catalog_path)
        df = df[df["splits"] == splits]
        return [
            f"{self.dataset_dir}/{splits}/{fname}" for fname in df["fnames"].tolist()
        ]

    def preprocess_fn(self, image_path):
        # Define a wrapper function to load and preprocess the .npy file
        def _load_npy(image_path):
            image = np.load(image_path.decode("utf-8")).astype(np.float32)
            image = np.transpose(image, (1, 2, 0))
            image = (image - self.mean) / self.std
            return image

        image = tf.numpy_function(_load_npy, [image_path], tf.float32)
        image.set_shape(
            [None, None, self.image_channels]
        )  # Set shape to (H, W, C) where C is 4

        image = tf.image.resize(image, self.input_image_size)
        return image

    def load_dataset(self, split):
        file_paths = glob.glob(f"{self.dataset_dir}/{split}/*.npy")
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)

        # split to sequences
        dataset = dataset.map(self.preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return tfds.as_numpy(dataset)

    def infinite_generator(self, dataset):
        while True:
            data_iter = iter(dataset)
            try:
                while True:
                    yield next(data_iter)
            except StopIteration:
                continue

    def train_dataloader(self):
        dataset = self.load_dataset("train")
        return self.infinite_generator(dataset)

    def val_dataloader(self):
        dataset = self.load_dataset("val")
        return self.infinite_generator(dataset)
