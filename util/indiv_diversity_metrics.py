import numpy as np
import math
from enum import Enum
from niapy.util.distances import euclidean


__all__ = ["IDT", "ISI", "IFMea", "IFMed"]


class IndivDiversityMetric(Enum):
    IDT = "idt"
    ISI = "isi"
    IFMea = "ifmea"
    IFMed = "ifmed"


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
            distances[p] += euclidean(populations[t].population[p], populations[t + 1].population[p])

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
        # calculate euclidean distance between positions in first and last iteration
        d = euclidean(populations[0].population[p], populations[len(populations) - 1].population[p])

        if d != 0:
            isi[p] /= d

    return isi


def IFMea(populations, pop_size):
    r"""Individual Fitness Mean.

    Args:
        populations (numpy.ndarray[PopulationData]): populations.
        pop_size (int): population size

    Returns:
        numpy.ndarray: Array of IFMea values.
    """
    sums = np.zeros(pop_size)
    for t in range(len(populations) - 1):
        sums = np.add(sums, populations[t].population_fitness)

    return sums / len(populations)


def IFMed(populations):
    r"""Individual Fitness Median.

    Args:
        populations (numpy.ndarray[PopulationData]): populations.

    Returns:
        numpy.ndarray: Array of IFMed values.
    """
    fitness_values = []
    for t in range(len(populations) - 1):
        fitness_values.append(populations[t].population_fitness)

    return np.median(fitness_values, axis=0)
