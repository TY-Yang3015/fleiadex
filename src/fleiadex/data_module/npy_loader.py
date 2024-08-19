import numpy as np
import hydra
import tensorflow as tf
from einops import rearrange
import json
import logging

import numpy as np
from PIL import Image
import math
import jax.numpy as jnp
import os


def max_depth(directory):
    max_depth = 0
    for root, dirs, _ in os.walk(directory):
        # Calculate the current depth by counting the separators in the path
        current_depth = root.count(os.sep) - directory.count(os.sep)
        if current_depth > max_depth:
            max_depth = current_depth
    return max_depth


class FleiadexDataLoader:
    """
    the dataloader for ``.npy`` format dataset.

    :var data_dir: the data directory, must be a string.
    :var batch_size: the batch size.
    :var fixed_normalisation_spec: only specify if you pre-computed the mean and std and want to normalise the
        data based on these values. dimension must match channel number.
    :var pre_split: bool. only set to ``True`` if you organised train and validation dataset in train and val folders.
    :var rescale_max: the max value of rescaling. only applies when both rescale_max and rescale_max are set.
    :var rescale_min: the min value of rescaling. only applies when both rescale_max and rescale_min are set.
    :var validation_size: the validation set size. when ``pre_split`` is set to ``True``, then this will be ignored.
    :var sequenced: whether the loaded data should be sequenced.
    :var sequence_length: the length of sequence if sequenced is True.
    :var auto_normalisation: whether to automatically normalise the data on the channel axis. the axis will
                            be selected based on the ``layout`` parameter.
    :var target_layout: the layout of the target dataset. only support ``'h w c'`` and ``'c h w'``.
                        not case-sensitive.
    :var output_image_size: the output image size, not this not dependent on the input image size.

    """

    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            fixed_normalisation_spec: tuple[list] | None = None,
            pre_split: bool = False,
            image_channels: int = 4,
            rescale_max: float | None = None,
            rescale_min: float | None = None,
            validation_size: float = 0.2,
            sequenced: bool = False,
            sequence_length: int = 1,
            auto_normalisation: bool = True,
            target_layout: str = "h w c",
            output_image_size: int = 128,
            *args,
            **kwargs,
    ):
        self.auto_normalisation = auto_normalisation
        self.pre_split = pre_split
        self.fixed_normalisation_spec = None

        if fixed_normalisation_spec is not None:
            self.auto_normalisation = False
            self.fixed_normalisation_spec = jnp.array(list(fixed_normalisation_spec))
            logging.info('fixed normalisation data spec received. auto-normalisation disabled.')
            logging.info(f'channel-wise mean: {self.fixed_normalisation_spec[0]}')
            logging.info(f'channel-wise std: {self.fixed_normalisation_spec[1]}')

        def reshape_element(element):
            shape = tf.shape(element)

            if len(shape) == 4:
                reshaped = tf.reshape(element, [-1, shape[1], shape[2], shape[3]])
                return reshaped
            else:
                return element

        def load_npy_file(file_path):
            data = np.load(file_path)

            return reshape_element(data.astype(np.float32))

        if data_dir.endswith(".npy"):
            self.data = tf.data.Dataset.from_tensor_slices(jnp.load(data_dir))
            logging.info(f"loaded single .npy file {data_dir}.")
        else:
            if self.pre_split:
                logging.info(f"searching for split train val data from {data_dir}.")
                self.train_dir = data_dir + "/train"
                self.validation_dir = data_dir + "/val"

                if (max_depth(self.train_dir) != 0) or (max_depth(self.validation_dir) != 0):
                    raise ValueError(f"split data directory should not contain any subfolders.")
                else:
                    npy_files_dirs_train = []
                    for subdir in os.listdir(self.train_dir):
                        if str(subdir).endswith(".npy"):
                            npy_files_dirs_train.append(self.train_dir + '/' + subdir)

                    self.train_data = tf.data.Dataset.from_tensor_slices(npy_files_dirs_train)
                    self.train_data = self.train_data.map(
                        lambda x: tf.numpy_function(load_npy_file, [x], [tf.float32]),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )

                npy_files_dirs_val = []
                for subdir in os.listdir(self.validation_dir):
                    if str(subdir).endswith(".npy"):
                        npy_files_dirs_val.append(self.validation_dir + '/' + subdir)

                self.validation_data = tf.data.Dataset.from_tensor_slices(npy_files_dirs_val)
                self.validation_data = self.validation_data.map(
                    lambda x: tf.numpy_function(load_npy_file, [x], [tf.float32]),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

            else:
                if max_depth(data_dir) != 0:
                    raise ValueError(f"data directory should not contain any subfolders."
                                     f" turn on by set pre_split=True if you intended to pass a dataset"
                                     f"with pre-defined train and val dataset.")
                else:
                    npy_files_dirs = []
                    for subdir in os.listdir(data_dir):
                        if str(subdir).endswith(".npy"):
                            npy_files_dirs.append(data_dir + '/' + subdir)

                    self.data = tf.data.Dataset.list_files(npy_files_dirs)
                    self.data = self.data.map(
                        lambda x: tf.numpy_function(load_npy_file, [x], [tf.float32]),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )

        self.data_dir = data_dir
        self.origin_shape = (self.data.cardinality().numpy() if not self.pre_split
                             else self.train_data.concatenate(self.validation_data).cardinality().numpy(),) + (
                                next(self.data.take(1).as_numpy_iterator())[0].shape if not self.pre_split
                                else next(self.train_data.take(1).as_numpy_iterator())[0].shape)
        self.batch_size = batch_size
        self.channels = image_channels
        self.rescale_max = rescale_max
        self.rescale_min = rescale_min
        self.validation_size = validation_size
        self.sequenced = sequenced
        if self.sequenced:
            self.sequence_length = sequence_length
        else:
            self.sequence_length = None

        self.target_layout = target_layout
        self.output_image_size = output_image_size

        if self.pre_split:
            self.train_data = self.train_data.map(lambda x: self._make_layout(x),
                                                  num_parallel_calls=tf.data.AUTOTUNE, )
            self.validation_data = self.validation_data.map(lambda x: self._make_layout(x),
                                                            num_parallel_calls=tf.data.AUTOTUNE, )
        else:
            self.data = self.data.map(lambda x: self._make_layout(x),
                                      num_parallel_calls=tf.data.AUTOTUNE, )

        if len(self.target_layout.split(" ")) != 3:
            raise ValueError(
                "target layout is only required for the last 3 channels."
            )

        self.channels = (int(next(self.data.take(1).as_numpy_iterator()).shape[-1]) if not self.pre_split
                         else int(next(self.train_data.take(1).as_numpy_iterator()).shape[-1]))

        if self.fixed_normalisation_spec is not None:
            assert self.channels == jnp.shape(self.fixed_normalisation_spec)[-1], ("channel number must"
                                                                                   "match the normalisation parameter"
                                                                                   "dimensions.")

        if self.target_layout.split(" ")[-1].casefold() == "c":
            if self.pre_split:
                self.mean, self.std = self._get_distribution_spec(self.train_data.concatenate(self.validation_data),
                                                                  (0, 1))
            else:
                self.mean, self.std = self._get_distribution_spec(self.data, (0, 1))
        else:
            if self.pre_split:
                self.mean, self.std = self._get_distribution_spec(self.train_data.concatenate(self.validation_data),
                                                                  (1, 2))
            else:
                self.mean, self.std = self._get_distribution_spec(self.data, (1, 2))

        if pre_split is not True:
            size = float(self.data.cardinality() if not pre_split else (self.train_data.cardinality()
                                                                        + self.validation_data.cardinality()))

            if self.validation_size == 0:
                raise ValueError("validation size must be non-zero.")
            else:
                self.validation_length = int(self.validation_size * size)

        if self.pre_split:
            self.data_max = self.train_data.concatenate(self.validation_data).reduce(0.,
                                                                                     lambda x, y: float(tf.reduce_max(
                                                                                         tf.stack([tf.reduce_max(x),
                                                                                                   tf.reduce_max(y)]))
                                                                                     ))
            self.data_min = self.train_data.concatenate(self.validation_data).reduce(0.,
                                                                                     lambda x, y: float(tf.reduce_min(
                                                                                         tf.stack([tf.reduce_min(x),
                                                                                                   tf.reduce_min(
                                                                                                       y)]))

                                                                                     ))
        else:
            self.data_max = self.data.reduce(0., lambda x, y: float(tf.reduce_max(
                tf.stack([tf.reduce_max(x),
                          tf.reduce_max(
                              y)]))))
            self.data_min = self.data.reduce(0., lambda x, y: float(tf.reduce_min(
                tf.stack([tf.reduce_min(x),
                          tf.reduce_min(
                              y)]))))

        self.__processed__ = False

    def _make_layout(self, data: jnp.ndarray) -> jnp.ndarray:
        # check the last three dims (C H W) / (H W C)
        shape = self.origin_shape[-3:]
        if len(shape) == 2:
            data = tf.expand_dims(data, axis=-1)
            shape = data.shape[-3:]

        data = tf.ensure_shape(data, [None, None, None])
        if shape[0] == shape[1]:
            if self.target_layout.split(" ")[-1].casefold() == "c":
                pass
            else:
                data = rearrange(data, "w h c -> c w h")
        elif shape[1] == shape[2]:
            if self.target_layout.split(" ")[0].casefold() == "c":
                pass
            else:
                data = rearrange(data, "c w h -> w h c")
        elif shape[0] != shape[1] != shape[2]:
            raise ValueError("shape mismatch, only " "square images are supported.")
        else:
            raise ValueError(
                "unable to make layout, "
                "please set target_layout to "
                "None and manually adjust the"
                "layout."
            )
        return data

    def _get_distribution_spec(self, dataset: tf.data.Dataset, reduced_axis: tuple[int, int]):
        """
        calculate mean and std dynamically for large dataset.

        $$s^2 = \frac{\sum s_i^2 + (\mu_i - \mu)^2}{N}$$

        **this equation assumes a large sample size. (sampling ddof not taken into account)**
        """

        element_var = dataset.map(lambda x: tf.math.square(tf.math.reduce_std(x, axis=reduced_axis)))
        element_mean = dataset.map(lambda x: tf.math.reduce_mean(x, axis=reduced_axis))

        size = float(dataset.cardinality())

        overall_mean = element_mean.reduce(jnp.zeros(self.channels), lambda x, y: x + y) / size

        overall_var = element_var.reduce(jnp.zeros(self.channels), lambda x, y: x + y) / size

        mean_dist = element_mean.map(lambda x: tf.math.square(x - overall_mean)).reduce(jnp.zeros(self.channels),
                                                                                        lambda x, y: x + y)
        mean_dist /= size

        overall_std = tf.math.sqrt(mean_dist + overall_var)

        return overall_mean, overall_std

    def _preprocess(self, data: jnp.ndarray) -> jnp.ndarray:

        if self.auto_normalisation:
            data = (data - self.mean) / self.std
        elif self.fixed_normalisation_spec is not None:
            data = (data - self.fixed_normalisation_spec[0]) / self.fixed_normalisation_spec[1]

        if self.target_layout.split(" ")[-1].casefold() == "c":
            data = tf.image.resize(
                data,
                (
                    self.output_image_size,
                    self.output_image_size,
                ),
                method="nearest",
            )
        else:
            data = tf.image.resize(
                data,
                (
                    data.shape[0],
                    data.shape[1],
                    self.output_image_size,
                    self.output_image_size,
                ),
                method="nearest",
            )

        if not ((self.rescale_max is None) and (self.rescale_min is None)):
            data -= self.data_min
            data /= self.data_max - self.data_min
            data *= self.rescale_max - self.rescale_min
            data += self.rescale_min

        return data

    def _train_val_split(self):
        if self.pre_split:
            pass
        else:
            self.validation_data = self.data.take(self.validation_length)
            self.train_data = self.data.skip(self.validation_length)

        return self

    def write_data_summary(self, save_dir: str | None = None):

        if save_dir is None:
            try:
                save_dir = (
                    save_dir
                ) = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            except Exception:
                logging.error("failed to save the dataset spec.")

        summary = {}
        summary["original_shape"] = [int(i) for i in self.origin_shape]
        processed_shape = (self.train_data.concatenate(
            self.validation_data).cardinality().numpy().astype(int) if self.pre_split else
                           self.data.cardinality().numpy().astype(int),) + ((next(
            self.train_data.take(1).as_numpy_iterator()).shape) if self.pre_split else
                                                                            next(self.data.take(
                                                                                1).as_numpy_iterator()).shape)
        summary["processed_shape"] = [int(i) for i in processed_shape]
        if self.auto_normalisation or (self.fixed_normalisation_spec is not None):
            summary["channel-wise_mean"] = list(self.mean.numpy().astype(float))
            summary["channel-wise_std"] = list(self.std.numpy().astype(float))
        if not self.pre_split:
            summary["validation_size"] = float(self.validation_size)
            summary["validation_length"] = int(self.validation_length)
        summary["data_max"] = float(self.data_max.numpy())
        summary["data_min"] = float(self.data_min.numpy())

        for key, value in summary.items():
            if isinstance(value, np.int64):
                summary[key] = int(value)
            if isinstance(value, np.float32):
                summary[key] = float(value)

        with open(f"{save_dir}/summary.json", "w") as f:
            json.dump(summary, f)

        return self

    def get_train_test_dataset(self):
        self._train_val_split()

        if not self.__processed__:
            train_dataset = self.train_data.map(lambda x: self._preprocess(x), num_parallel_calls=tf.data.AUTOTUNE)
            val_dataset = self.validation_data.map(lambda x: self._preprocess(x), num_parallel_calls=tf.data.AUTOTUNE)
            self.__processed__ = True
        else:
            train_dataset = self.train_data
            val_dataset = self.validation_data

        train_dataset = train_dataset.cache()
        val_dataset = val_dataset.cache()

        if self.sequenced:
            self.train_dataset = train_dataset.batch(
                self.sequence_length, drop_remainder=True
            )
            self.train_dataset = self.train_dataset.batch(
                self.batch_size, drop_remainder=True
            )
            # self.train_dataset = val_dataset.rebatch(self.batch_size,
            #                                         drop_remainder=True)

            self.val_dataset = val_dataset.batch(
                self.sequence_length, drop_remainder=True
            )
            self.val_dataset = self.val_dataset.batch(
                self.batch_size, drop_remainder=True
            )
            # self.val_dataset = val_dataset.rebatch(self.batch_size,
            #                                       drop_remainder=True)

        else:
            self.train_dataset = train_dataset.batch(
                self.batch_size, drop_remainder=True
            )
            self.val_dataset = val_dataset.batch(self.batch_size, drop_remainder=True)

        self.train_dataset = (
            self.train_dataset.shuffle(10000)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .repeat()
        )
        self.val_dataset = (
            self.val_dataset.shuffle(10000)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .repeat()
        )

        return (
            self.train_dataset.as_numpy_iterator(),
            self.val_dataset.as_numpy_iterator(),
        )

    def reverse_preprocess(self, image: jnp.ndarray) -> jnp.ndarray:

        if self.target_layout.split(" ")[-1].casefold() == "c":
            image = image * self.std + self.mean
        elif self.target_layout.split(" ")[0].casefold() == "c":
            if self.sequenced:
                image = rearrange(image, "b t c h w -> b t h w c")
            else:
                image = rearrange(image, "b c h w -> b h w c")

            image = image * self.std + self.mean

            if self.sequenced:
                image = rearrange(image, "b t h w c -> b c t h w")
            else:
                image = rearrange(image, "b h w c -> b c h w")

        if not ((self.rescale_max is None) and (self.rescale_min is None)):
            image -= self.rescale_min
            image /= self.rescale_max
            image *= self.data_max - self.data_min
            image += self.data_min

        return image

    def get_complete_dataset(
            self,
            batched: bool = False,
            sequenced: bool = False,
            repeat: bool = False,
            as_iterator: bool = True,
    ):
        if self.pre_split:
            self.data = self.train_data.concatenate(self.validation_data)

        if not self.__processed__:
            self.data = self.data.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = self.data.cache()

        if sequenced and batched:
            dataset = dataset.batch(self.sequence_length, drop_remainder=True).batch(
                self.batch_size, drop_remainder=True
            )
        elif batched and not sequenced:
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
        elif not batched and not sequenced:
            pass
        else:
            raise ValueError("batched must be True for sequencing.")

        dataset = dataset.shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)

        if repeat:
            dataset = dataset.repeat()

        if as_iterator:
            return dataset.as_numpy_iterator()
        else:
            return dataset

#dataloader = FleiadexDataLoader(
#    data_dir='/home/arezy/Desktop/satel_clean/',
#    pre_split=True,
#    fixed_normalisation_spec=([264.5569, 249.6994, 228.8998, 242.5480],
#                              [25.1830, 18.3450, 7.5322, 13.1226]),
#    batch_size=10,
#    output_image_size=16
#)

#print(next(dataloader.get_train_test_dataset()[0]).shape)
#print(dataloader.origin_shape)
