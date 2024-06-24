import jax.numpy as jnp


def linear_schedule(step: int, min_val: float, max_val: float) -> jnp.ndarray:
    return jnp.linspace(min_val, max_val, step, dtype=jnp.float32)
