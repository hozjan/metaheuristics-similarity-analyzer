import numpy as np
import math
from typing import Any
from niapy.algorithms import Algorithm
from niapy.util.factory import (
    _algorithm_options,
    get_algorithm,
)

__all__ = ["random_float_with_step", "smape"]


def random_float_with_step(low, high, step, size=None, replace=True):
    steps = np.arange(low / step, high / step)
    random_steps = np.random.choice(steps, size=size, replace=replace)
    random_floats = random_steps * step
    return random_floats


def smape(first: np.ndarray, second: np.ndarray):
    """calculates 1-SMAPE between two arrays.
        Arrays must have the same length.

    Args:
        first (np.ndarray): first array.
        second (np.ndarray): second array.

    Returns:
        1-smape (float): 1-SMAPE value.
    """

    return 1.0 - np.mean(
        np.abs((first - second)) / (np.abs(first) + np.abs(second) + math.ulp(0.0))
    )


def get_algorithm_by_name(name: str | Algorithm, *args, **kwargs):
    """Get an instance of the algorithm by name. If string it must be listed in niapy's `_algorithm_options` method.

    Args:
        name (str | Algorithm): String name of the algorithm class or the class itself.

    Returns:
        algorithm (Algorithm): An instance of the algorithm.
    """

    if not isinstance(name, str) and issubclass(name, Algorithm):
        return name(*args, **kwargs)
    elif isinstance(name, str) and name not in _algorithm_options():
        raise KeyError(
            f"Could not find algorithm by name `{name}` in the niapy library."
        )
    else:
        return get_algorithm(name, *args, **kwargs)
