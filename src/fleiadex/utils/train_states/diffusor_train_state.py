import jax
from flax.training import train_state


class DiffusorTrainState(train_state.TrainState):
    key: jax.Array
    consts: jax.Array
