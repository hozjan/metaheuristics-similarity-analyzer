import numpy as np
from msa.tools.optimization_data import PopulationData, PopDiversityMetric
from niapy.problems import Problem
from niapy.util.distances import euclidean
import itertools

__all__ = ['DPC']


class DPC(PopDiversityMetric):
    r"""Distance to Population Centroid.

    Reference paper:
        Ursem, Rasmus. (2002). Diversity-Guided Evolutionary Algorithms. 2439. 10.1007/3-540-45712-7_45.

    """

    def __init__(self, problem: Problem, *args, **kwargs):
        """Initialize Distance to Population Centroid.

        Args:
            problem (Problem): Optimization problem.

        """
        super().__init__(*args, **kwargs)
        self.problem = problem

    def _evaluate(self, popData: PopulationData, *args, **kwargs):
        if popData.population is None:
            return 0

        P, N = np.shape(popData.population)
        L = euclidean(self.problem.upper, self.problem.lower)
        avg_point = np.mean(popData.population, axis=0)
        distances = np.linalg.norm(popData.population - list(itertools.repeat(avg_point, P)), axis=1)
        dpc = np.sum(distances, axis=0)

        return dpc / (P * L)
