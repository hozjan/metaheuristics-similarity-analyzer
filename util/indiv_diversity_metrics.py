import numpy as np
import math
from enum import Enum
from niapy.util.distances import euclidean
import copy


__all__ = ["IDT", "ISI", "IFMea", "IFMed"]


class IndivDiversityMetric(Enum):
    IDT = "idt"
    ISI = "isi"
    IFMea = "ifmea"
    IFMed = "ifmed"


def IDT(populations, pop_size):
    r"""Individual Distance Traveled.

    Args:
        populations (numpy.ndarray[PopulationData]): Populations.
        pop_size (int): Population size.

    Returns:
        numpy.ndarray: Array of IDT values.

    """
    distances = []
    for t in range(len(populations) - 1):
        first = np.array([pop for pop in populations[t].population])
        second = np.array([pop for pop in populations[t + 1].population])
        distances.append(np.linalg.norm(first - second, axis = 1))
    
    distances = np.sum(distances, axis=0)

    return distances


def ISI(populations, pop_size, return_idt=False):
    r"""Individual Sinuosity Index. Utilizes Individual Distance Traveled function.

    Args:
        populations (numpy.ndarray[PopulationData]): populations.
        pop_size (int): population size.
        return_idt (Optional[bool]): Also return Individual Distance Traveled.

    Returns:
        numpy.ndarray: Array of ISI values and also IDT values based on arguments.
    """
    isi = IDT(populations, pop_size)
    idt = copy.deepcopy(isi) if return_idt else None

    for p in range(pop_size):
        # calculate euclidean distance between positions in first and last iteration
        d = euclidean(
            populations[0].population[p],
            populations[len(populations) - 1].population[p],
        )

        if d != 0:
            isi[p] /= d

    return isi, idt if return_idt else isi


def IFMea(populations, pop_size):
    r"""Individual Fitness Mean.

    Args:
        populations (numpy.ndarray[PopulationData]): Populations.
        pop_size (int): Population size.

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
        populations (numpy.ndarray[PopulationData]): Populations.

    Returns:
        numpy.ndarray: Array of IFMed values.
    """
    fitness_values = []
    for t in range(len(populations) - 1):
        fitness_values.append(populations[t].population_fitness)

    return np.median(fitness_values, axis=0)
