import tensorflow as tf
import numpy as np
from omegaconf import DictConfig
from typing import Tuple
from absl import logging
import os


def load_dataset(config: DictConfig) -> Tuple[iter, iter]:
    dataset_dir = config.data_spec.dataset_dir
    validation_split = config.data_spec.validation_split
    image_height = image_width = config.data_spec.image_size
    batch_size = config.hyperparams.batch_size

    def preprocess(image):
        image = ((tf.cast(image, tf.float32) / 255) * (config.data_spec.clip_max - config.data_spec.clip_min)
                 + config.data_spec.clip_min)
        logging.info(f'image data is clipped to [{config.data_spec.clip_min}, {config.data_spec.clip_max}].')
        return image

    if dataset_dir.split('.')[-1] == 'npy':
        return handle_npy_dataset(dataset_dir,
                                  validation_split,
                                  image_height,
                                  image_width,
                                  batch_size,
                                  preprocess)
    elif os.path.exists(dataset_dir):
        return handle_dir_dataset(dataset_dir,
                                  validation_split,
                                  image_height,
                                  image_width,
                                  batch_size,
                                  preprocess)
    else:
        raise ValueError(f'dataset dir or file {dataset_dir} does not exist')


def adjust_dataset_shape(data: np.ndarray) -> np.ndarray:
    if len(data.shape) != 4:
        raise ValueError('data should have 4 dimensions.')

    if np.argmin(data.shape) != 3:

        if np.argmin(data.shape) == 1:
            new_data = np.zeros((data.shape[0], data.shape[2], data.shape[3], data.shape[1]))
            for i, v in enumerate(data):
                for j, w in enumerate(v):
                    new_data[i, :, :, j] = data[i, j, :, :]
        else:
            raise ValueError(f'check the dimension of data {data.shape}')

        logging.info(f'data shape: {data.shape} is adjusted to {new_data.shape}')

        return new_data
    else:
        return data


def handle_dir_dataset(dataset_dir: str,
                       validation_split: float,
                       image_height: int,
                       image_width: int,
                       batch_size: int,
                       fn:callable) -> Tuple[iter, iter]:
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=(image_height, image_width),
        batch_size=batch_size
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=(image_height, image_width),
        batch_size=batch_size
    )

    train_dataset = train_dataset.map(fn).cache().shuffle(1000).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).repeat().as_numpy_iterator()
    validation_dataset = validation_dataset.map(fn).cache().prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).repeat().as_numpy_iterator()

    return train_dataset, validation_dataset


def handle_npy_dataset(dataset_dir: str,
                       validation_split: float,
                       image_height: int,
                       image_width: int,
                       batch_size: int,
                       fn: callable) -> Tuple[iter, iter]:
    dataset = np.load(dataset_dir)
    dataset_size = dataset.shape[0]

    dataset = adjust_dataset_shape(dataset)

    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    train_size = int(dataset_size * (1 - validation_split))
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)

    train_dataset = train_dataset.map(fn).cache().shuffle(1000).batch(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).repeat().as_numpy_iterator()
    validation_dataset = validation_dataset.map(fn).cache().batch(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).repeat().as_numpy_iterator()

    return train_dataset, validation_dataset
