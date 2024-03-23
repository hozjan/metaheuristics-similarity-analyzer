import numpy as np
import math
from enum import Enum
from niapy.problems import Problem

__all__ = ["PDC", "PED", "PMD"]


class DiversityMetric(Enum):
    PDC = "pdc"
    PED = "ped"
    PMD = "pmd"


def PDC(population, problem: Problem):
    r"""Distance to Population Centroid.

    Reference paper:
        Ursem, Rasmus. (2002). Diversity-Guided Evolutionary Algorithms. 2439. 10.1007/3-540-45712-7_45.

    Args:
        population (numpy.ndarray): population.
        problem (Problem): Optimization problem.

    Returns:
        PDC value.

    """
    P, N = np.shape(population)
    L = 0
    pdc = 0

    for lb, ub in zip(problem.lower, problem.upper):
        L += math.pow(ub - lb, 2)
    L = math.sqrt(L)

    avg_point = np.zeros(N)
    for p in population:
        for j, x in enumerate(p):
            avg_point[j] += x

    avg_point /= P

    for p in population:
        sum = 0
        for j, x in enumerate(p):
            sum += math.pow(x - avg_point[j], 2)
        pdc += math.sqrt(sum)

    return pdc / (P * L)


def PED(population):
    r"""Population Euclidean distance.

    Args:
        population (numpy.ndarray): population.

    Returns:
        PED value.

    """
    ped = 0

    for index_i, pi in enumerate(population):
        for pj in population[index_i + 1 :]:
            sum = 0
            for xi, xj in zip(pi, pj):
                sum += math.pow(xi - xj, 2)
            ped += math.sqrt(sum)

    return ped


def PMD(population):
    r"""Population Manhattan distance.

    Args:
        population (numpy.ndarray): population.

    Returns:
        PMD value.

    """
    pmd = 0

    for index_i, pi in enumerate(population):
        for pj in population[index_i + 1 :]:
            sum = 0
            for xi, xj in zip(pi, pj):
                sum += abs(xi - xj)
            pmd += sum

    return pmd
