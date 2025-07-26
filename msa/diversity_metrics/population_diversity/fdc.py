from niapy.util.array import full_array
import numpy as np
import itertools
from msa.tools.optimization_data import PopulationData, PopDiversityMetric
from niapy.problems import Problem

__all__ = ["FDC"]


class FDC(PopDiversityMetric):
    r"""Fitness Distance Correlation.

    Reference paper:
        Jones, T.C. & Forrest, S. (1995). Fitness Distance Correlation as a
        Measure of Problem Difficulty for Genetic Algorithms.

    """

    def __init__(
        self, problem: Problem, global_optimum: list[float], extend_global_optimum: bool = False, *args, **kwargs
    ):
        r"""Initialize Fitness Distance Correlation.

        Args:
            problem (Problem): Optimization problem.
            global_optimum (list[float]): Location of the global optimum of the provided problem. Length must match
                `problem.dimension` property if `extend_global_optimum` is False.
            extend_global_optimum (Optional[bool]): Extend `global_optimum` list to match `problem.dimension` by
                repeating first value of the list.

        """
        super().__init__(*args, **kwargs)
        self.problem = problem

        if extend_global_optimum and len(global_optimum) >= 1:
            self.global_optimum = full_array(global_optimum[0], problem.dimension)
        elif problem.dimension == len(global_optimum):
            self.global_optimum = global_optimum
        else:
            raise ValueError(
               """`global_optimum` dimension must match Problem dimension or at least have length of 1 and utilize
               `extend_global_optimum` argument."""
            )

    def _evaluate(self, popData: PopulationData, *args, **kwargs):
        if (
            not isinstance(self.problem, Problem)
            or popData.population is None
            or popData.population_fitness is None
        ):
            return 0

        P, N = np.shape(popData.population)
        D = np.linalg.norm(popData.population - list(itertools.repeat(self.global_optimum, P)), axis=1)
        f_avg = popData.population_fitness.mean()
        f_std = popData.population_fitness.std()
        d_avg = D.mean()
        d_std = D.std()

        CFD = sum((popData.population_fitness - f_avg) * (D - d_avg) / P)

        if f_std != 0.0 and d_std != 0:
            FDC = CFD / (f_std * d_std)
        else:
            FDC = 0.0

        return FDC + 1.0
