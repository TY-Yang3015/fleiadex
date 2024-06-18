import numpy as np


def linear_schedule(step: int, min_val: float, max_val: float) -> np.ndarray:
    return np.linspace(min_val, max_val, step, dtype=np.float32)
