import numpy as np
import hydra
import tensorflow as tf
from einops import rearrange
import json
import logging
import os
import numpy as np
from PIL import Image
import math
import jax.numpy as jnp


def save_image(data_loader,
               save_dir: str,
               ndarray: jnp.ndarray,
               fp: any,
               nrow: int = 8,
               padding: int = 2,
               pad_value: float = 0.0,
               format_img: any = None):
    """Make a grid of images and Save it into an image file.

    Args:
        data_loader (DataLoader): the dataloader object.
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
        raise TypeError(f'array_like of tensors expected, got {type(ndarray)}')

    ndarray = jnp.asarray(ndarray)

    if data_loader.rescale_min is None or data_loader.rescale_max is None:
        ndarray -= data_loader.data_min
        ndarray /= data_loader.data_max
    else:
        ndarray -= data_loader.rescale_min
        ndarray /= data_loader.rescale_max

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
                   y * height + padding: (y + 1) * height,
                   x * width + padding: (x + 1) * width,
                   ].set(ndarray[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = np.array(jnp.clip(grid * 255, 0, 255).astype(jnp.uint8))
    im = Image.fromarray(ndarr.copy(), mode='RGB')
    im.save(save_dir + fp, format=format_img)


def max_depth(directory):
    max_depth = 0
    for root, dirs, _ in os.walk(directory):
        # Calculate the current depth by counting the separators in the path
        current_depth = root.count(os.sep) - directory.count(os.sep)
        if current_depth > max_depth:
            max_depth = current_depth
    return max_depth


class NpyLoaderCore:
    """
    the data loader backend for ``.npy`` format dataset.

    :var data_dir: the data directory, must be a string.
    :var batch_size: the batch size.
    :var dir_sort_key: function that returns the key used to sort the directory order.
    :var rescale_max: the max value of rescaling. only applies when both rescle_max and rescale_max are set.
    :var rescale_min: the min value of rescaling. only applies when both rescle_max and rescale_min are set.
    :var validation_size: the validation set size.
    :var sequenced: whether the loaded data should be sequenced.
    :var sequence_length: the length of sequence if sequenced is True.
    :var auto_normalisation: whether to automatically normalise the data on the channel axis. the axis will
                            be selected based on the ``layout`` parameter.
    :var auto_layout_adjustment: whether to automatically adjust the data on the channel axis.
    :var target_layout: the layout of the target dataset. only support ``'h w c'`` and ``'c h w'``.
                        not case-sensitive.
    :var output_image_size: the output image size, this is not dependent on the input image size.

    """

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 dir_sort_key: callable = None,
                 rescale_max: float | None = None,
                 rescale_min: float | None = None,
                 validation_size: float = 0.2,
                 sequenced: bool = False,
                 sequence_length: int = 1,
                 auto_normalisation: bool = True,
                 auto_layout_adjustment: bool = True,
                 target_layout: str = 'h w c',
                 output_image_size: int = 128):

        if data_dir.endswith('.npy'):
            self.dataset = tf.data.Dataset.from_tensor_slices(jnp.load(data_dir))
            logging.info(f'loaded single .npy file {data_dir}.')
        else:
            if max_depth(data_dir) != 0:
                raise ValueError(f'data directory should not contain any subfolders.')
            else:
                npy_files_dirs = []
                for subdir in os.listdir(data_dir):
                    if str(subdir).endswith(".npy"):
                        npy_files_dirs.append(subdir)
                if dir_sort_key is not None:
                    npy_files_dirs.sort(key=lambda x: dir_sort_key(x))
                    self.dataset = tf.data.Dataset.list_files(npy_files_dirs)
                    self.dataset = tf.data.Dataset.map(lambda x: jnp.load(data_dir + '/' + x),
                                                       num_parallel_calls=tf.data.AUTOTUNE)

                else:
                    self.dataset = tf.data.Dataset.list_files(npy_files_dirs)
                    self.dataset = tf.data.Dataset.map(lambda x: jnp.load(data_dir + '/' + x),
                                                       num_parallel_calls=tf.data.AUTOTUNE)

        if len(next(self.dataset.as_numpy_iterator()).shape) != 3:
            raise ValueError(f'only image or image-like data is supported, which should have 3 dimensions.')

        self.target_layout = target_layout
        self.data_shape = jnp.array(jnp.shape(next(self.dataset.as_numpy_iterator())))
        logging.info(f'received element data shape is {self.data_shape}')
        if auto_layout_adjustment:
            self.dataset = self.dataset.map(
                lambda x: self._make_layout(x)
            )
        else:
            pass

        self.data_size = self.dataset.reduce(0, lambda x, _: x + 1)
        logging.info(f'received total data size is {self.data_size}')

        self.batch_size = batch_size
        self.rescale_max = rescale_max
        self.rescale_min = rescale_min
        self.validation_size = validation_size
        self.sequenced = sequenced
        self.sequence_length = sequence_length
        self.auto_normalisation = auto_normalisation
        self.output_image_size = output_image_size

    def _make_layout(self, element):
        # check the last three dims (C H W) / (H W C)
        if jnp.argmin(self.data_shape) == 2:
            if (self.target_layout.split(' ')[-1]
                    == 'c'.casefold()):
                pass
            else:
                element = rearrange(element,
                                    'w h c -> c w h')
        elif jnp.argmin(self.data_shape) == 0:
            if (self.target_layout.split(' ')[0]
                    == 'c'.casefold()):
                pass
            else:
                element = rearrange(element,
                                    'c w h -> w h c')
        else:
            raise ValueError('unable to make layout, '
                             'please set target_layout to '
                             'None and manually adjust the '
                             'layout.')
        return element

    def _preprocess(self):
        self.mean = self.dataset.map(lambda x: jnp.mean(x, axis=(0, 1)))
        self.std = self.dataset.map(lambda x: jnp.std(x, axis=(0, 1)))

        return self

    def _train_val_split(self):
        pass

    def write_data_summary(self,
                           save_dir: str | None = None):

        if save_dir is None:
            try:
                save_dir = save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            except Exception:
                logging.error('failed to save the dataset spec.')

        summary = {}
        summary['original_shape'] = self.origin_shape
        summary['processed_shape'] = self.data.shape
        summary['channel-wise_mean'] = self.mean.tolist()
        summary['channel-wise_std'] = self.std.tolist()
        summary['validation_size'] = self.validation_size
        summary['validation_length'] = self.validation_length
        summary['data_max'] = self.data_max
        summary['data_min'] = self.data_min

        with open(f'{save_dir}/summary.json', 'w') as f:
            json.dump(summary, f)

        return self

    def read_data_summary(self, json_dir: str):
        with open(json_dir, 'r') as f:
            summary = json.load(f)

        self.mean = np.array(summary['channel-wise_mean'])
        self.std = np.array(summary['channel-wise_std'])
        self.validation_size = np.array(summary['validation_size'])
        self.validation_length = np.array(summary['validation_length'])
        self.data_max = np.array(summary['data_max'])
        self.data_min = np.array(summary['data_min'])

        logging.info('dataset summary read successfully.')
        return self

    def get_train_test_dataset(self):
        pass

    def reverse_preprocess(self, image):
        pass

    def save_image(self,
                   save_dir,
                   ndarray,
                   fp,
                   nrow=8,
                   padding=2,
                   pad_value=0.0,
                   format_img=None):

        return save_image(self, save_dir, ndarray,
                          fp, nrow=nrow,
                          padding=padding,
                          pad_value=pad_value,
                          format_img=format_img)

    def get_complete_dataset(self,
                             batched: bool = False,
                             sequenced: bool = False,
                             repeat: bool = False,
                             as_iterator: bool = True):
        pass


dl = NpyLoaderCore(
    data_dir='../exp_data/satel_array_202312bandopt00_clear.npy',
    batch_size=1
)

print(next(dl.dataset.batch(10).as_numpy_iterator()).shape)
