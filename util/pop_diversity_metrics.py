import numpy as np
import math
from enum import Enum
from niapy.problems import Problem
from niapy.util.distances import euclidean
import itertools

__all__ = ["PDC", "PED", "PMD", "AAD", "PDI", "FDC", "PFSD", "PFMea", "PFMed"]


class PopDiversityMetric(Enum):
    PDC = "pdc"
    PED = "ped"
    PMD = "pmd"
    AAD = "aad"
    PDI = "pdi"
    FDC = "fdc"
    PFSD = "pfsd"
    PFMea = "pfmea"
    PFMed = "pfmed"


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

    L = euclidean(problem.upper, problem.lower)

    avg_point = np.mean(population, axis=0)

    distances = np.linalg.norm(
        population - list(itertools.repeat(avg_point, P)), axis=1
    )
    pdc = np.sum(distances, axis=0)

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
            ped += euclidean(pi, pj)
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
            ad += euclidean(p, pi)
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
    _population = np.copy(population)

    # m - number of individuals
    # n - number of dimensions
    m, n = np.shape(_population)

    # expected distance between any two individuals in an uniform distribution over [0, 1]^n
    a_n = math.pow(1 / m, 1 / n) * math.sqrt(n)
    omega = -math.log(epsilon) / a_n
    sigma = -math.log(m) / math.log(0.01)

    # normalizing values to [0, 1]
    for pi in range(m):
        for xi in range(n):
            _population[pi][xi] = (_population[pi][xi] - problem.lower[xi]) / (
                problem.upper[xi] - problem.lower[xi]
            )

    # calculate numerator part of the pdi equation
    sum = 0
    for xi in _population:
        # average similarity of xi to members of population
        p_hat = 0
        for xj in _population:
            # calculate euclidean distance
            d = euclidean(xi, xj)
            p_hat += math.exp(-omega * d) / m

        sum += math.log(math.pow(p_hat, sigma))

    return -sum / (m * math.log(m))


def FDC(population, population_fitness, problem: Problem):
    r"""Fitness Distance Correlation.

    Reference paper:
        Jones, T.C. & Forrest, S. (1995). Fitness Distance Correlation as a Measure of Problem Difficulty for Genetic Algorithms.

    Args:
        population (numpy.ndarray): population.
        population_fitness (numpy.ndarray): population fitness.
        problem (Problem): Optimization problem.

    Returns:
        FDC value.
    """
    if problem.global_optimum is None:
        return 0.0

    P, N = np.shape(population)
    D = np.linalg.norm(
        population - list(itertools.repeat(problem.global_optimum, P)), axis=1
    )
    f_avg = population_fitness.mean()
    f_std = population_fitness.std()
    d_avg = D.mean()
    d_std = D.std()

    CFD = sum((population_fitness - f_avg) * (D - d_avg) / P)

    if f_std != 0.0 and d_std != 0:
        FDC = CFD / (f_std * d_std)
    else:
        FDC = 0.0

    return FDC


def PFSD(population_fitness):
    r"""Population Fitness Standard Deviation.

    Args:
        population_fitness (numpy.ndarray): population fitness.

    Returns:
        PFSD value.
    """
    return population_fitness.std()


def PFMea(population_fitness):
    r"""Population Fitness Mean.

    Args:
        population_fitness (numpy.ndarray): population fitness.

    Returns:
        PFMea value.
    """
    return population_fitness.mean()


def PFMed(population_fitness):
    r"""Population Fitness Median.

    Args:
        population_fitness (numpy.ndarray): population fitness.

    Returns:
        PFMed value.
    """
    return np.median(population_fitness)
