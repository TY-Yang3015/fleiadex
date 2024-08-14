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


def save_image(
    data_loader,
    save_dir: str,
    ndarray: jnp.ndarray,
    fp: any,
    nrow: int = 8,
    padding: int = 2,
    pad_value: float = 0.0,
    format_img: any = None,
):
    """Make a grid of images and Save it into an image file.

    Args:
        data_loader (LegacyDataLoader): the dataloader object.
        save_dir (str): the directory to save the image file.
        ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
        fp:  A filename(string) or file object
        nrow (int, optional): Number of images displayed in each row of the grid.
          The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
        format_img(Optional):  If omitted, the format to use is determined from the
          filename extension. If a file object was used instead of a filename,
          this parameter should always be used.
    """

    if not (
        isinstance(ndarray, jnp.ndarray)
        or (
            isinstance(ndarray, list)
            and all(isinstance(t, jnp.ndarray) for t in ndarray)
        )
    ):
        raise TypeError(f"array_like of tensors expected, got {type(ndarray)}")

    ndarray = jnp.asarray(ndarray)

    if data_loader.rescale_min is None or data_loader.rescale_max is None:
        ndarray -= data_loader.data_min
        ndarray *= data_loader.data_max - data_loader.data_min
    else:
        ndarray -= data_loader.rescale_min
        ndarray *= data_loader.rescale_max - data_loader.rescale_min

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)
    elif ndarray.ndim == 4 and ndarray.shape[-1] > 3:
        ndarray = ndarray[:, :, :, :3]

    # adjust intensity for visualisation purpose
    ndarray *= 2

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = (
        int(ndarray.shape[1] + padding),
        int(ndarray.shape[2] + padding),
    )
    num_channels = ndarray.shape[3]
    grid = jnp.full(
        (height * ymaps + padding, width * xmaps + padding, num_channels),
        pad_value,
    ).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ].set(ndarray[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = np.array(jnp.clip(grid * 255, 0, 255).astype(jnp.uint8))
    im = Image.fromarray(ndarr.copy(), mode="RGB")
    im.save(save_dir + fp, format=format_img)


class LegacyDataLoader:
    """
    the dataloader for ``.npy`` format dataset.

    :var data_dir: the data directory, must be a string.
    :var batch_size: the batch size.
    :var rescale_max: the max value of rescaling. only applies when both rescle_max and rescale_max are set.
    :var rescale_min: the min value of rescaling. only applies when both rescle_max and rescale_min are set.
    :var validation_size: the validation set size.
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
        image_channels: int = 4,
        rescale_max: float | None = None,
        rescale_min: float | None = None,
        validation_size: float = 0.2,
        sequenced: bool = False,
        sequence_length: int = 1,
        auto_normalisation: bool = True,
        target_layout: str = "h w c",
        image_size: int = 128,
        *args,
        **kwargs,
    ):

        self.data_dir = data_dir
        self.data = np.load(data_dir)
        self.origin_shape = self.data.shape
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

        self.auto_normalisation = auto_normalisation
        self.target_layout = target_layout
        self.output_image_size = image_size

        self._make_layout()

        if len(self.target_layout.split(" ")) != 3:
            raise ValueError(
                "target layout is only required for the last" "3 channels."
            )

        if self.target_layout.split(" ")[-1] == "c".casefold():
            self.std = np.std(self.data, axis=(0, 1, 2), ddof=1)
            self.mean = np.mean(self.data, axis=(0, 1, 2))
        else:
            self.std = np.std(self.data, axis=(0, 2, 3), ddof=1)
            self.mean = np.mean(self.data, axis=(0, 2, 3))

        size = len(self.data)

        if self.validation_size == 0:
            raise ValueError("validation size must" "non-zero.")
        else:
            self.validation_length = int(self.validation_size * size)

        self.data_max = np.max(self.data)
        self.data_min = np.min(self.data)

        self.__processed__ = False

    def _make_layout(self):
        # check the last three dims (C H W) / (H W C)
        shape = np.array(np.shape(self.data))[-3:]
        if shape[0] == shape[1]:
            if self.target_layout.split(" ")[-1] == "c".casefold():
                pass
            else:
                self.data = rearrange(self.data, "b w h c -> b c w h")
        elif shape[1] == shape[2]:
            if self.target_layout.split(" ")[0] == "c".casefold():
                pass
            else:
                self.data = rearrange(self.data, "b c w h -> b w h c")
        elif shape[0] != shape[1] != shape[2]:
            raise ValueError("shape mismatch, only " "square images are supported.")
        else:
            raise ValueError(
                "unable to make layout, "
                "please set target_layout to "
                "None and manually adjust the"
                "layout."
            )
        return self

    def _preprocess(self):

        if self.__processed__:
            return self

        if self.auto_normalisation:
            self.data = (self.data - self.mean) / self.std

        if self.target_layout.split(" ")[-1] == "c".casefold():
            self.data = tf.image.resize(
                self.data,
                (
                    self.output_image_size,
                    self.output_image_size,
                ),
                method="nearest",
            )
        else:
            self.data = tf.image.resize(
                self.data,
                (
                    self.data.shape[0],
                    self.data.shape[1],
                    self.output_image_size,
                    self.output_image_size,
                ),
                method="nearest",
            )

        if not ((self.rescale_max is None) or (self.rescale_min is None)):
            self.data -= self.data_min
            self.data /= self.data_max - self.data_min
            self.data *= self.rescale_max - self.rescale_min
            self.data += self.rescale_min
            logging.info(
                f"dataset rescaled to [{np.max(self.data)}, {np.min(self.data)}]"
            )
        else:
            logging.info("no scaling was applied.")

        self.__processed__ = True

        return self

    def _train_val_split(self):
        self.validation_data = self.data[: self.validation_length]
        self.train_data = self.data[self.validation_length :]

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
        summary["original_shape"] = self.origin_shape
        summary["processed_shape"] = self.data.shape
        summary["channel-wise_mean"] = self.mean.tolist()
        summary["channel-wise_std"] = self.std.tolist()
        summary["validation_size"] = self.validation_size
        summary["validation_length"] = self.validation_length
        summary["data_max"] = self.data_max
        summary["data_min"] = self.data_min

        with open(f"{save_dir}/summary.json", "w") as f:
            json.dump(summary, f)

        return self

    def read_data_summary(self, json_dir: str):
        with open(json_dir, "r") as f:
            summary = json.load(f)

        self.mean = np.array(summary["channel-wise_mean"])
        self.std = np.array(summary["channel-wise_std"])
        self.validation_size = np.array(summary["validation_size"])
        self.validation_length = np.array(summary["validation_length"])
        self.data_max = np.array(summary["data_max"])
        self.data_min = np.array(summary["data_min"])

        logging.info("dataset summary read successfully.")
        return self

    def get_train_test_dataset(self):
        self._preprocess()._train_val_split()

        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_data)
        val_dataset = tf.data.Dataset.from_tensor_slices(self.validation_data)

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

    def reverse_preprocess(self, image):

        if self.target_layout.split(" ")[-1] == "c".casefold():
            image = image * self.std + self.mean
        elif self.target_layout.split(" ")[0] == "c".casefold():
            if self.sequenced:
                image = rearrange(image, "b t c h w -> b t h w c")
            else:
                image = rearrange(image, "b c h w -> b h w c")

            image = image * self.std + self.mean

            if self.sequenced:
                image = rearrange(image, "b t h w c -> b c t h w")
            else:
                image = rearrange(image, "b h w c -> b c h w")

        if not ((self.rescale_max is None) or (self.rescale_min is None)):
            image -= self.rescale_min
            image /= self.rescale_max
            image *= self.data_max - self.data_min
            image += self.data_min

        return image

    def save_image(
        self, save_dir, ndarray, fp, nrow=8, padding=2, pad_value=0.0, format_img=None
    ):

        return save_image(
            self,
            save_dir,
            ndarray,
            fp,
            nrow=nrow,
            padding=padding,
            pad_value=pad_value,
            format_img=format_img,
        )

    def get_complete_dataset(
        self,
        batched: bool = False,
        sequenced: bool = False,
        repeat: bool = False,
        as_iterator: bool = True,
    ):
        self._preprocess()

        dataset = tf.data.Dataset.from_tensor_slices(self.data).cache()

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


# dataloader = DataLoader(
#    data_dir='/home/arezy/Desktop/fleiadex/src/fleiadex/exp_data/satel_array_202312bandopt00_clear.npy',
#    batch_size=10,
#    sequenced=True,
#    sequence_length=12,
#    output_image_size=16
# )

# print(next(dataloader.get_train_test_dataset()[0]).shape)
