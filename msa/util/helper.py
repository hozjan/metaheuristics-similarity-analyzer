import numpy as np
import numpy.typing as npt
import math
import time
from niapy.algorithms.algorithm import Algorithm
from niapy.util.factory import (
    _algorithm_options,
    get_algorithm,
)

__all__ = ["random_float_with_step", "smape", "get_algorithm_by_name", "timer"]


def random_float_with_step(low, high, step, size=None, replace=True):
    steps = np.arange(low / step, high / step)
    random_steps = np.random.choice(steps, size=size, replace=replace)
    random_floats = random_steps * step
    return random_floats


def smape(first: npt.NDArray, second: npt.NDArray):
    """calculates 1-SMAPE between two arrays.
        Arrays must have the same length.

    Args:
        first (numpy.ndarray): first array.
        second (numpy.ndarray): second array.

    Returns:
        1-smape (float): 1-SMAPE value.
    """

    return 1.0 - np.mean(np.abs((first - second)) / (np.abs(first) + np.abs(second) + math.ulp(0.0)))


def get_algorithm_by_name(name: str | Algorithm, *args, **kwargs):
    """Get an instance of the algorithm by name. If string it must be listed in niapy's `_algorithm_options` method.

    Args:
        name (str | Algorithm): String name of the algorithm class or the class itself.

    Returns:
        algorithm (Algorithm): An instance of the algorithm.

    Raises:
        KeyError: Provided algorithm not found in the niapy framework.
    """

    if not isinstance(name, str) and issubclass(name, Algorithm):
        return name(*args, **kwargs)
    elif isinstance(name, str) and name not in _algorithm_options():
        raise KeyError(f"Could not find algorithm by name `{name}` in the niapy framework.")
    else:
        return get_algorithm(name, *args, **kwargs)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
