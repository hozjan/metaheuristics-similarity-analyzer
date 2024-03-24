import numpy as np
import math
from enum import Enum
from niapy.problems import Problem

__all__ = ["PDC", "PED", "PMD", "AAD"]


class DiversityMetric(Enum):
    PDC = "pdc"
    PED = "ped"
    PMD = "pmd"
    AAD = "aad"


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
        for x, avg in zip(p, avg_point):
            sum += math.pow(x - avg, 2)
        pdc += math.sqrt(sum)

    return pdc / (P * L)


def PED(population):
    r"""Population Euclidean Distance.

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
    r"""Population Manhattan Distance.

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


def AAD(population):
    r"""Average of the Average Distance around all Particles in the Swarm.

    Reference paper:
        O. Olorunda and A. P. Engelbrecht, "Measuring exploration/exploitation in particle swarms using swarm diversity," China, 2008, pp. 1128-1134, doi: 10.1109/CEC.2008.4630938.

    Args:
        population (numpy.ndarray): population.

    Returns:
        AAD value.

    """
    P, N = np.shape(population)
    aad = 0

    for pi in population:
        ad = 0
        for p in population:
            sum = 0
            for x, xi in zip(p, pi):
                sum += math.pow(xi - x, 2)
            ad += math.sqrt(sum)
        aad += ad / P

    return aad / P
