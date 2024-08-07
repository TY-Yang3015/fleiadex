import flax.linen as nn
import jax.numpy as jnp
import jax


class PositionalEmbedding(nn.Module):
    """
    the positional embedder with layers to learn.

    :cvar input_shape: the specified input shape. **must be 4d!** for 5d input, give the last 4 dims.
    :cvar mode: the embedding mode. only ``t+h+w`` is supported at the moment.
    """

    input_shape: tuple[int, int, int, int]
    mode: str = 't+h+w'

    def setup(self) -> None:
        if (self.input_shape is None) or len(self.input_shape) != 4:
            raise ValueError('input must have four dimensions.')

        self.t_dim, self.h_dim, self.w_dim, self.embed_dim = self.input_shape
        self.t_index = jnp.arange(self.t_dim)
        self.h_index = jnp.arange(self.h_dim)
        self.w_index = jnp.arange(self.w_dim)

        if self.mode == 't+h+w':
            self.t_embed = nn.Embed(num_embeddings=self.t_dim, features=self.embed_dim)
            self.h_embed = nn.Embed(num_embeddings=self.h_dim, features=self.embed_dim)
            self.w_embed = nn.Embed(num_embeddings=self.w_dim, features=self.embed_dim)
        else:
            raise NotImplementedError('only the default t+h+w is supported.')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        :param x: the input jax array. **must be 4d or 5d**.
        :return: the output with positional embedding added to the last four dimensions.
        """
        if self.mode == 't+h+w':
            if x.ndim == 4:
                embedding = (x + self.t_embed(self.t_index).reshape(self.t_dim, 1, 1, self.embed_dim)
                             * self.h_embed(self.h_index).reshape(1, self.h_dim, 1, self.embed_dim)
                             * self.w_embed(self.w_index).reshape(1, 1, self.w_dim, self.embed_dim))
            elif x.ndim == 5:
                embedding = (x + self.t_embed(self.t_index).reshape(self.t_dim, 1, 1, self.embed_dim)
                             * self.h_embed(self.h_index).reshape(1, self.h_dim, 1, self.embed_dim)
                             * self.w_embed(self.w_index).reshape(1, 1, self.w_dim, self.embed_dim))
            else:
                raise ValueError('input must be 4d or 5d.')
            return embedding


#rng = jax.random.PRNGKey(0)
#input_shape = (10, 5, 32, 32, 256)
#print(PositionalEmbedding(input_shape[1:]).tabulate(rng, jnp.zeros(input_shape, dtype=jnp.float32), console_kwargs={'width':150}))
