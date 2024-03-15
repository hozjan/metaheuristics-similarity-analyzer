import numpy as np
import math
from enum import Enum

__all__ = [
    'PDC'
]

class DiversityMetric(Enum):
    PDC = 'pdc'

def PDC(population):
    r"""Calculate the Distance to Population Centroid diversity metric.

    Args:
        population (numpy.ndarray): population.

    Returns:
        PDC value.

    """
    P, N = np.shape(population)
    pdc = 0
    centroid = np.zeros(N)
    for p in population:
        for j, x in enumerate(p):
            centroid[j] += x

    centroid = centroid / P

    for p in population:
        for j, x in enumerate(p):
            pdc += math.pow(x - centroid[j], 2)

    return pdc / P
