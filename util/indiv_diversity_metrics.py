import numpy as np
import math
from enum import Enum


__all__ = ["IDT", "ISI"]


class IndivDiversityMetric(Enum):
    IDT = "idt"
    ISI = "isi"


def IDT(populations, pop_size):
    r"""Individual Distance Traveled.

    Args:
        populations (numpy.ndarray[PopulationData]): populations.
        pop_size (int): population size

    Returns:
        numpy.ndarray: Array of IDT values.

    """
    distances = np.zeros(pop_size)
    for t in range(len(populations) - 1):
        for p in range(pop_size):
            # calculate euclidean distance
            euclidean_sum = 0
            for _xi, _xj in zip(populations[t].population[p], populations[t + 1].population[p]):
                euclidean_sum += math.pow(_xi - _xj, 2)
            d = math.sqrt(euclidean_sum)
            distances[p] += d

    return distances


def ISI(populations, pop_size):
    r"""Individual Sinuosity Index.

    Args:
        populations (numpy.ndarray[PopulationData]): populations.
        pop_size (int): population size

    Returns:
        numpy.ndarray: Array of ISI values.
    """
    isi = IDT(populations, pop_size)

    for p in range(pop_size):
        # calculate euclidean distance of first and last line
        euclidean_sum = 0
        for _xi, _xj in zip(populations[0].population[p], populations[len(populations)-1].population[p]):
            euclidean_sum += math.pow(_xi - _xj, 2)
        d = math.sqrt(euclidean_sum)
        isi[p] /= d

    return isi
