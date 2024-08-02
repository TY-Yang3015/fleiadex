import jax
from flax.training import train_state


class TrainStateWithBatchStats(train_state.TrainState):
    batch_stats: any
