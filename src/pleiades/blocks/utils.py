import flax.linen as nn
import jax.numpy as jnp
import jax


def pad_input(x: jnp.ndarray, pad_t, pad_h, pad_w, padding_type) -> jnp.ndarray:
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    if padding_type not in ['zeros', 'auto', 'edge', 'nearest']:
        raise NotImplementedError('unsupported padding type.')

    batch_size, t, h, w, c = x.shape

    if padding_type == 'auto' or 'zeros':
        return jnp.pad(x, ((0, 0), (0, pad_t), (0, pad_h), (0, pad_w), (0, 0)),
                       mode='constant', constant_values=0)
    elif padding_type == 'edge':
        return jnp.pad(x, ((0, 0), (0, pad_t), (0, pad_h), (0, pad_w), (0, 0)),
                       mode='edge')
    elif padding_type == 'nearest':
        return jax.image.resize(x, (batch_size, t+pad_t, h+pad_h, w+pad_w, c), method='nearest')


def unpad_output(x: jnp.ndarray, pad_t, pad_h, pad_w, padding_type) -> jnp.ndarray:
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    if padding_type not in ['zeros', 'auto', 'edge']:
        raise NotImplementedError('unsupported padding type.')

    batch_size, t, h, w, c = x.shape

    return x[:, :(t - pad_t), :(h - pad_h), :(w - pad_w), :]
