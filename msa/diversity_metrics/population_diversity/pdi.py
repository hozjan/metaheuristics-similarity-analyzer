import numpy as np
from msa.tools.optimization_data import PopulationData, PopDiversityMetric
from niapy.util.distances import euclidean
from niapy.problems import Problem
import math

__all__ = ['PDI']


class PDI(PopDiversityMetric):
    r"""Population Diversity Index.

    Reference paper:
        Smit, S.K. & Szlávik, Zoltán & Eiben, A.. (2011). Population diversity index:
        A new measure for population diversity. 269-270. 10.1145/2001858.2002010.

    """

    def __init__(self, problem: Problem, epsilon=0.001, *args, **kwargs):
        r"""Initialize Population Diversity Index.

        Args:
            problem (Problem): Optimization problem.
            epsilon (float): scaling parameter in exclusive range (0, 1).

        """
        super().__init__(*args, **kwargs)
        self.problem = problem
        self.epsilon = epsilon

    def _evaluate(self, popData: PopulationData, *args, **kwargs):
        if popData.population is None:
            return 0
        _population = np.copy(popData.population)

        # m - number of individuals
        # n - number of dimensions
        m, n = np.shape(_population)

        # expected distance between any two individuals in an uniform distribution over [0, 1]^n
        a_n = math.pow(1 / m, 1 / n) * math.sqrt(n)
        omega = -math.log(self.epsilon) / a_n
        sigma = -math.log(m) / math.log(0.01)

        # normalizing values to [0, 1]
        for pi in range(m):
            for xi in range(n):
                _population[pi][xi] = (_population[pi][xi] - self.problem.lower[xi]) / (
                    self.problem.upper[xi] - self.problem.lower[xi]
                )

        # calculate numerator part of the pdi equation
        sum = 0.0
        for xi in _population:
            # average similarity of xi to members of population
            p_hat = 0.0
            for xj in _population:
                # calculate euclidean distance
                d = euclidean(xi, xj)
                p_hat += math.exp(-omega * d) / m

            sum += math.log(math.pow(p_hat, sigma))

        return -sum / (m * math.log(m))
