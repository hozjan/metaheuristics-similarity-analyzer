import numpy as np
import scipy
from enum import Enum
from niapy.util.distances import euclidean
import copy
import scipy.stats

__all__ = ["IDT", "ISI", "IFM", "IFIQR", "IndivDiversityMetric"]


class IndivDiversityMetric(Enum):
    IDT = "IDT"
    ISI = "ISI"
    IFM = "IFM"
    IFIQR = "IFIQR"


def IDT(populations: list):
    r"""Individual Distance Traveled.

    Args:
        populations (numpy.ndarray[PopulationData]): Populations.

    Returns:
        numpy.ndarray: Array of IDT values.

    """
    distances = []
    for t in range(len(populations) - 1):
        first = np.array([pop for pop in populations[t].population])
        second = np.array([pop for pop in populations[t + 1].population])
        distances.append(np.linalg.norm(first - second, axis=1))

    distances = np.sum(distances, axis=0)

    return distances


def ISI(
    populations: list, return_idt=False
):
    r"""Individual Sinuosity Index. Utilizes Individual Distance Traveled function.

    Args:
        populations (numpy.ndarray[PopulationData]): populations.
        return_idt (Optional[bool]): Also return Individual Distance Traveled.

    Returns:
        numpy.ndarray: Array of ISI values and also IDT values based on arguments.
    """
    isi = IDT(populations)
    idt = copy.deepcopy(isi)

    for p in range(len(populations[0].population)):
        # calculate euclidean distance between positions in first and last iteration
        d = euclidean(
            populations[0].population[p],
            populations[len(populations) - 1].population[p],
        )

        if d != 0:
            isi[p] /= d

    if return_idt:
        return isi, idt
    else:
        return isi


def IFM(populations: list):
    r"""Individual Fitness Mean.

    Args:
        populations (numpy.ndarray[PopulationData]): Populations.

    Returns:
        numpy.ndarray: Array of IFM values.
    """
    sums = np.zeros(len(populations[0].population))
    for t in range(len(populations) - 1):
        sums = np.add(sums, populations[t].population_fitness)

    return sums / len(populations)


def IFIQR(populations: list):
    r"""Individual Fitness Interquartile Range.

    Args:
        populations (numpy.ndarray[PopulationData]): Populations.

    Returns:
        numpy.ndarray: Array of IFIQR values.
    """
    fitness_values = []
    for t in range(len(populations) - 1):
        fitness_values.append(populations[t].population_fitness)

    return np.array(scipy.stats.iqr(fitness_values, axis=0).tolist())
