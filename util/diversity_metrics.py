import numpy as np
import math
from enum import Enum
from niapy.problems import Problem

__all__ = ["PDC", "PED", "PMD", "AAD", "PDI"]


class DiversityMetric(Enum):
    PDC = "pdc"
    PED = "ped"
    PMD = "pmd"
    AAD = "aad"
    PDI = "pdi"


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


def PDI(population, problem: Problem, epsilon=0.001):
    r"""Population Diversity Index.

    Reference paper:
        Smit, S.K. & Szlávik, Zoltán & Eiben, A.. (2011). Population diversity index: A new measure for population diversity. 269-270. 10.1145/2001858.2002010.

    Args:
        population (numpy.ndarray): population.
        problem (Problem): Optimization problem.
        epsilon (float): scaling parameter in exclusive range (0, 1)

    Returns:
        PDI value.

    """
    # m - number of individuals
    # n - number of dimensions
    m, n = np.shape(population)

    # expected distance between any two individuals in an uniform distribution over [0, 1]^n
    a_n = math.pow(1 / m, 1 / n) * math.sqrt(n)
    omega = -math.log(epsilon) / a_n
    sigma = -math.log(m)/math.log(0.01)

    # normalizing values to [0, 1]
    for pi in range(m):
        for xi in range(n):
            population[pi][xi] = (population[pi][xi] - problem.lower[xi]) / (
                problem.upper[xi] - problem.lower[xi]
            )

    # calculate numerator part of the pdi equation
    sum = 0
    for xi in population:
        # average similarity of xi to members of population
        p_hat = 0
        for xj in population:
            # calculate euclidean distance
            euclidean_sum = 0
            for _xi, _xj in zip(xi, xj):
                euclidean_sum += math.pow(_xi - _xj, 2)
            d = math.sqrt(euclidean_sum)
            p_hat += math.exp(-omega * d) / m

        sum += math.log(math.pow(p_hat, sigma))

    return -sum / (m * math.log(m))
