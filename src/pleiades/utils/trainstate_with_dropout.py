import jax
from flax.training import train_state


class TrainStateWithDropout(train_state.TrainState):
    key: jax.Array
