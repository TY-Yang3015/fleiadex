import numpy as np
from PIL import Image
import math
import jax.numpy as jnp


def save_image(
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
    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)
    elif ndarray.ndim == 4 and ndarray.shape[-1] == 2:  # double-channel images
        ndarray = jnp.concatenate((ndarray[..., 0], ndarray[..., 1], ndarray[..., 1]), -1)
    elif ndarray.ndim == 4 and ndarray.shape[-1] > 3:
        ndarray = ndarray[:, :, :, :3]

    ndarray = jnp.asarray(ndarray)
    ndarray -= ndarray.min()
    ndarray /= ndarray.max()
    ndarray *= 1.

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
    im = Image.fromarray(ndarr.copy(), mode="RGB")
    im.save(save_dir + fp, format=format_img)


def save_many_images(
        save_dir: str,
        ndarray_list: list[jnp.ndarray, ...],
        fp: any,
        nrow: int = 8,
        padding: int = 2,
        pad_value: float = 0.0,
        format_img: any = None,
):
    """Make a grid of images and Save it into an image file.

    Args:
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

    new_array = []
    for i, ndarray in enumerate(ndarray_list):
        if not (
                isinstance(ndarray, jnp.ndarray)
                or (
                        isinstance(ndarray, list)
                        and all(isinstance(t, jnp.ndarray) for t in ndarray)
                )
        ):
            raise TypeError(f"array_like of tensors expected, got {type(ndarray)}")


        if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
            ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)
        elif ndarray.ndim == 4 and ndarray.shape[-1] == 2:  # double-channel images
            ndarray = jnp.concatenate((ndarray[..., 0], ndarray[..., 1], ndarray[..., 1]), -1)
        elif ndarray.ndim == 4 and ndarray.shape[-1] > 3:
            ndarray = ndarray[:, :, :, :3]

        new_array.append(ndarray)

    ndarray = jnp.concatenate(new_array, 0)

    ndarray = jnp.asarray(ndarray)
    ndarray -= ndarray.min()
    ndarray /= ndarray.max()
    ndarray *= 1.

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
    im = Image.fromarray(ndarr.copy(), mode="RGB")
    im.save(save_dir + fp, format=format_img)
