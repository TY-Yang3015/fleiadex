import tensorflow as tf
import ml_collections as mc
from typing import Tuple
import tensorflow_datasets as tfds


def load_dataset(config: mc.ConfigDict) -> Tuple[iter, iter]:
    dataset_dir = config.data_spec.dataset_dir
    VALIDATION_SPLIT = config.data_spec.validation_split
    IMG_HEIGHT = IMG_WIDTH = config.data_spec.image_size
    BATCH_SIZE = config.hyperparams.batch_size

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=42,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=42,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    def preprocess(image, shit):
        image = tf.image.resize(image, (128, 128))
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        return image

    train_dataset = train_dataset.map(preprocess).cache().shuffle(1000).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).repeat().as_numpy_iterator()
    validation_dataset = validation_dataset.map(preprocess).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat().as_numpy_iterator()


    return train_dataset, validation_dataset

