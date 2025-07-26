import numpy as np
import scipy
from msa.tools.optimization_data import SingleRunData, IndivDiversityMetric

__all__ = ['IFIQR']


class IFIQR(IndivDiversityMetric):
    r"""Individual Fitness Interquartile Range."""

    def __init__(self, *args, **kwargs):
        r"""Initialize Individual Fitness Interquartile Range."""
        super().__init__(*args, **kwargs)

    def _evaluate(self, srd: SingleRunData, *args, **kwargs):
        fitness_values = []
        for t in range(len(srd.populations) - 1):
            fitness_values.append(srd.populations[t].population_fitness)

        return np.array(scipy.stats.iqr(np.array(fitness_values), axis=0).tolist())
