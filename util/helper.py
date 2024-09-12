import numpy as np
import math

__all__ = ["random_float_with_step", "s_mape"]


def random_float_with_step(low, high, step, size=None, replace=True):
    steps = np.arange(low / step, high / step)
    random_steps = np.random.choice(steps, size=size, replace=replace)
    random_floats = random_steps * step
    return random_floats


def s_mape(first: np.ndarray, second: np.ndarray):
    """calculates S-MAPE between two arrays.
        Arrays must have the same length.

    Args:
        first (np.ndarray): first array.
        second (np.ndarray): second array.

    Returns:
        s-mape (float): S-MAPE value.
    """

    return np.mean(
        np.abs((first - second)) / (np.abs(first) + np.abs(second) + math.ulp(0.0))
    )
