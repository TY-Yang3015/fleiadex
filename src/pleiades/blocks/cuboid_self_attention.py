import flax.core.nn.attention
import flax.linen as nn
import jax.numpy as jnp
import jax
from einops import rearrange

from src.pleiades.errors import StructureError
from src.pleiades.blocks.utils import pad_input, unpad_output


class FinalProjection(nn.Module):
    dimension: int
    drop_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = nn.Dense(self.dimension)(x)
        x = nn.Dropout(self.drop_rate, deterministic=not train)(x)
        return x


class AttentionDropout(nn.Module):
    drop_rate: float

    @nn.compact
    def __call__(self, x_input: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = nn.Dropout(self.drop_rate, deterministic=not train)(x_input)
        return x


class CuboidSelfAttention(nn.Module):
    """
    the implementation of cuboid attention mechanism in the paper **EarthFormer**
    (https://arxiv.org/abs/2207.05833). You are advised to read the paper before
    using this layer.

    takes a 4d tensor with the shape ``(batch_size, time_length, width, height, channels)``
    (**Important: this shape is not the same order as the original implementation** due to
    the difference between the default convolutional kernels in ``pytorch`` and ``tensorflow``/
    ``flax.linen``) as input, reshape it to ``(batch_size, cuboids, cuboid_volume, channels)``
    before multi-head attention. It's believed that this re-shuffling will improve the
    ability of the transformer to capture inter-regional (spatial/temporal) correlation.

    :cvar attention_heads: int. number of attention heads in the multi-head attention mechanism.
        this must be a factor of input channels (i.e. divisible).
    :cvar input_channels: int. the number of input channels.
    :cvar cuboid_size: 3d tuple. the size of the cuboids in the order
        ``(cuboid_time_length, cuboid_width, cuboid_height)``. padding will be applied if
        relative dimensions are not divisible by cuboid_size before shuffling.
    :cvar shift_size: 3d tuple. the shift applied to the cuboids when shuffling. any shift larger
        than the cuboid sizes will be regarded as cycling.
    :cvar strategy: 3d tuple of str. ``l`` for "local" or ``d`` for "dilated". "local" means gathering the
        adjacent pixels as a cuboid, whereas "dilated" means mixing pixels from different
        region and time with gap up_round(dimension/cuboid_size). Check out the paper for
        details.
    :cvar padding_type: str. the padding type to use for padding the input tensors.
    :cvar qkv_bias: bool, whether to use bias for qkv matrices.
    :cvar attention_dropout_rate: float, dropout rate for attention block.
    :cvar use_relative_position: bool, whether to use relative position for attentions.
    :cvar use_final_projector: bool, whether to use a final projection layer.
    :cvar final_projection_dropout: float, dropout rate for final projection layer. only used
        if the use_final_projector is True.

    """

    attention_heads: int = 4
    input_channels: int = 4
    cuboid_size: tuple[int, int, int] = (2, 7, 7)
    shift_size: tuple[int, int, int] = (0, 0, 0)
    strategy: tuple[str, str, str] = ('l', 'l', 'l')
    padding_type: str = 'auto'
    qkv_bias: bool = False
    attention_dropout_rate: float = 0.1
    use_relative_position: bool = True
    use_final_projector: bool = True
    final_projection_dropout: float = 0.1

    def setup(self):
        self.layer_norm = nn.LayerNorm()

        if self.use_relative_position:
            self.relative_position_bias_table = self.param(
                'relative_position_bias_table', nn.initializers.truncated_normal(stddev=0.02),
                (jnp.int32((2 * self.cuboid_size[0] - 1) * (2 * self.cuboid_size[1] - 1) * (2 * self.cuboid_size[2] - 1)),
                 self.attention_heads))
            relative_position_index = self._get_relative_position_index()
            key = self.make_rng('constants')
            self.relative_position_index = self.variable('constants',
                                                         'relative_position_index',
                                                         nn.initializers.normal(),
                                                         key, relative_position_index.shape)
            self.relative_position_index = relative_position_index

        if self.use_final_projector:
            self.final_projector = FinalProjection(self.input_channels, self.final_projection_dropout)

        self.qkv = nn.Dense(self.input_channels * 3, use_bias=self.qkv_bias)
        self.attention_dropout = AttentionDropout(self.attention_dropout_rate)

    def _get_relative_position_index(self) -> jnp.ndarray:
        t_coords = jnp.arange(self.cuboid_size[0])
        h_coords = jnp.arange(self.cuboid_size[1])
        w_coords = jnp.arange(self.cuboid_size[2])
        coords = jnp.stack(jnp.meshgrid(h_coords, t_coords, w_coords))

        coords = rearrange(coords, 'b t h w -> b (t h w)')
        relative_coords = (rearrange(coords, 'b thw -> b thw 1')
                           - rearrange(coords, 'b thw -> b 1 thw'))
        relative_coords = rearrange(relative_coords, 'b thw1 thw2 -> thw1 thw2 b')

        relative_coords.at[:, :, 0].set(relative_coords[:, :, 0] + self.cuboid_size[0] - 1)
        relative_coords.at[:, :, 1].set(relative_coords[:, :, 1] + self.cuboid_size[1] - 1)
        relative_coords.at[:, :, 2].set(relative_coords[:, :, 2] + self.cuboid_size[2] - 1)

        relative_coords.at[:, :, 0].set(relative_coords[:, :, 0] *
                                        (2 * self.cuboid_size[1] - 1) * (2 * self.cuboid_size[2] - 1))
        relative_coords.at[:, :, 1].set(relative_coords[:, :, 1] *
                                        (2 * self.cuboid_size[2] - 1))
        relative_position_index = relative_coords.sum(axis=-1)

        return relative_position_index

    def _cuboid_reorder(self, x: jnp.ndarray, cuboid_size, strategy) -> jnp.ndarray:
        batch_size, t, h, w, c = x.shape
        number_of_cuboid = t // cuboid_size[0] * h // cuboid_size[1] * w // cuboid_size[2]
        cuboid_volume = cuboid_size[0] * cuboid_size[1] * cuboid_size[2]
        intermediate_shape = []

        n_block_axis = []
        block_axis = []
        for i, (block_size, total_size, element_strategy) \
                in enumerate(zip(cuboid_size, (t, h, w), strategy)):
            if element_strategy == 'l':
                intermediate_shape.extend([int(total_size // block_size), int(block_size)])
                n_block_axis.append(2 * i + 1)
                block_axis.append(2 * i + 2)
            elif element_strategy == 'd':
                intermediate_shape.extend([int(block_size), int(total_size // block_size)])
                n_block_axis.append(2 * i + 2)
                block_axis.append(2 * i + 1)
            else:
                NotImplementedError(f"the elementwise strategy {element_strategy} is not implemented.\
                use 'l' or 'd' instead.")

        x = x.reshape((batch_size,) + tuple(intermediate_shape) + (c,))
        x = jnp.permute_dims(x, ((0,) + tuple(n_block_axis) + tuple(block_axis) + (7,)))
        x = x.reshape(batch_size, int(number_of_cuboid), int(cuboid_volume), c)
        return x

    def _reverse_cuboid_reorder(self, x: jnp.ndarray, cuboid_size, strategy, original_shape) -> jnp.ndarray:
        batch_size, number_of_cuboid, cuboid_volume, c = x.shape
        t, h, w = original_shape

        permutation_axis = [0]
        for i, (block_size, total_size, element_strategy) \
                in enumerate(zip(cuboid_size, (t, h, w), strategy)):
            if element_strategy == 'l':
                permutation_axis.append(i + 1)
                permutation_axis.append(i + 4)

            elif element_strategy == 'd':
                permutation_axis.append(i + 4)
                permutation_axis.append(i + 1)
            else:
                NotImplementedError(f"the elementwise strategy {element_strategy} is not implemented.\
                                    use 'l' or 'd' instead.")

        permutation_axis.append(7)
        x = x.reshape(batch_size, int(t // cuboid_size[0]), int(h // cuboid_size[1]),
                      int(w // cuboid_size[2]),
                      int(cuboid_size[0]), int(cuboid_size[1]), int(cuboid_size[2]), c)
        x = jnp.permute_dims(x, tuple(permutation_axis))
        x = x.reshape(batch_size, int(t), int(h), int(w), c)
        return x

    def _compute_attention_mask(self, pads, shape, size, shift, strategy, padding_type) -> jnp.ndarray:

        t, h, w = shape
        pad_t = pads[0]
        pad_h = pads[1]
        pad_w = pads[2]
        data_mask = None

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            if padding_type == 'auto':
                data_mask = jnp.ones((1, t, h, w, 1))
                data_mask = jnp.pad(data_mask, ((0, 0), (0, pad_t), (0, pad_h), (0, pad_w), (0, 0)))
        else:
            data_mask = jnp.ones((1, t, h, w, 1))

        if any(i > 0 for i in shift):
            if padding_type == 'auto':
                data_mask = jnp.roll(data_mask, shift=(-shift[0], -shift[1], -shift[2]), axis=(1, 2, 3))
        if padding_type == 'auto':
            data_mask = self._cuboid_reorder(data_mask, size, strategy)
            data_mask = data_mask.reshape(data_mask.shape[1], data_mask.shape[2])

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            shift_mask = jnp.zeros((1, t + pad_t, h + pad_h, w + pad_w, 1))
        else:
            shift_mask = jnp.zeros((1, t, h, w, 1))
        count = 0

        for ti in slice(-size[0]), slice(-size[0], -shift[0]), slice(-shift[0], None):
            for hi in slice(-size[1]), slice(-size[1], -shift[1]), slice(-shift[1], None):
                for wi in slice(-size[2]), slice(-size[2], -shift[2]), slice(-shift[2], None):
                    shift_mask = shift_mask.at[:, ti, hi, wi, :].set(count)
                    count += 1
        shift_mask = self._cuboid_reorder(shift_mask, size, strategy)
        shift_mask = shift_mask.reshape(data_mask.shape)
        attention_mask = jnp.expand_dims(shift_mask, axis=1) - jnp.expand_dims(shift_mask, axis=2) == 0
        if padding_type == 'auto':
            attention_mask = jnp.expand_dims(data_mask, axis=1) * jnp.expand_dims(data_mask, axis=2) * attention_mask

        return attention_mask

    def masked_softmax(self, attention_score, mask, axis=-1):
        def masked_fill(mask, a, fill):
            return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))
        if len(mask.shape) == 3:
            mask = jnp.expand_dims(mask, axis=0)
            mask = jnp.expand_dims(mask, axis=0)

        if mask is not None:
            mask = jnp.tile(mask, (attention_score.shape[0], attention_score.shape[1], 1, 1, 1))
            attention_score = masked_fill(jnp.logical_not(mask), attention_score, -1e18)
            attention_weights = nn.softmax(attention_score, axis=axis) * mask
        else:
            attention_weights = nn.softmax(attention_score, axis=axis)

        return attention_weights

    def __call__(self, x_input: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = self.layer_norm(x_input)

        batch_size, t, h, w, c_in = jnp.shape(x)
        assert c_in == self.input_channels, ("the input channel dimension "
                                             "does not much the specified dimension value.")
        assert c_in % self.attention_heads == 0, ("the attention head dimension must "
                                                  "be a factor of the input dimension.")

        pad_t = (self.cuboid_size[0] - t % self.cuboid_size[0]) % self.cuboid_size[0]
        pad_h = (self.cuboid_size[1] - h % self.cuboid_size[1]) % self.cuboid_size[1]
        pad_w = (self.cuboid_size[2] - w % self.cuboid_size[2]) % self.cuboid_size[2]
        x = (pad_input(x, pad_t, pad_h, pad_w, self.padding_type))

        if any(i > 0 for i in self.shift_size):
            shifted_x = jnp.roll(x, (-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                 axis=(1, 2, 3))
        else:
            shifted_x = x

        x_reordered = self._cuboid_reorder(shifted_x, cuboid_size=self.cuboid_size, strategy=self.strategy)
        _, number_of_cuboids, cuboid_volume, _ = x_reordered.shape

        attention_mask = self._compute_attention_mask((pad_t, pad_h, pad_w), (t, h, w), self.cuboid_size,
                                                      self.shift_size, self.strategy,
                                                      self.padding_type)

        channel_heads = c_in // self.attention_heads
        qkv = self.qkv(x_reordered).reshape(batch_size, number_of_cuboids, cuboid_volume, 3,
                                            self.attention_heads, channel_heads)

        qkv = rearrange(qkv, 'b noc cv qkv attn ch -> qkv b attn noc cv ch')
        # batch, number_of_cuboids, cuboid_volume, 3, attention_heads, channels_heads ->
        #           3, batch, attention_heads, number_of_cuboids, cuboid_volume, channel_heads
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * jnp.float32(self.input_channels // self.attention_heads) ** -.5  # scale q
        attention_score = q @ jnp.matrix_transpose(k)

        if self.use_relative_position:
            relative_position_bias = self.relative_position_bias_table[
                jnp.int32(self.relative_position_index[:cuboid_volume, :cuboid_volume])
            ].reshape(cuboid_volume, cuboid_volume, -1)
            relative_position_bias = rearrange(relative_position_bias, 'cv1 cv2 heads -> heads 1 cv1 cv2')
            attention_score += relative_position_bias

        # attention_score = nn.softmax(attention_score)
        attention_score = self.masked_softmax(attention_score, attention_mask)
        #print('before drop', attention_score.shape)
        attention_score = self.attention_dropout(attention_score, train)
        #print('before v', attention_score.shape)
        x_reordered = attention_score @ v
        #print('after_v', x_reordered.shape)
        x_reordered = rearrange(x_reordered, 'a b c d e -> a c d b e')
        x_reordered = x_reordered.reshape(batch_size, number_of_cuboids, cuboid_volume, c_in)

        if self.use_final_projector:
            x_reordered = self.final_projector(x_reordered, train)

        x_output = self._reverse_cuboid_reorder(x_reordered, self.cuboid_size, self.strategy,
                                                (t + pad_t, h + pad_h, w + pad_w))

        if any(i > 0 for i in self.shift_size):
            x_output = jnp.roll(x_output, self.shift_size, axis=(1, 2, 3))

        x_output = unpad_output(x_output, pad_t, pad_h, pad_w, self.padding_type)

        if x_output.shape != x_input.shape:
            raise StructureError(x_input.shape, x_output.shape)

        return x_output


#rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1), 'constants': jax.random.PRNGKey(2)}
#print(CuboidSelfAttention(input_channels=256,
#                          attention_heads=4).tabulate(rngs,
#                                                      jnp.zeros((10, 5, 16, 16, 256)), False,
#                                                      console_kwargs={'width': 150}, compute_flops=True
#                                                      ))
