import flax.linen as nn
import jax.numpy as jnp
import jax


class ObservationalMask(nn.Module):
    """
    masking the input by adding a channel. 0 indicates latent sample and 1 indicates
    encoded contextual data (e.g. encoded previous frames for image time series)

    :cvar input_length: the length of the contextual input sequence.
    """
    input_length: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        :param x: the input jax array with the shape ``(batch, ..., channel)``
        :return: the output jax array with the shape ``(batch, ..., channel + 1)``,
        where the added channel is the observational mask.
        """
        observation_indicator = jnp.ones_like(x[..., :1])
        observation_indicator.at[:, self.input_length:, ...].set(0.)
        return jnp.concatenate([x, observation_indicator], axis=-1)

#key = jax.random.PRNGKey(0)
#print(ObservationalMask(5).tabulate(key, jnp.zeros((10, 5, 64, 64, 256))))
