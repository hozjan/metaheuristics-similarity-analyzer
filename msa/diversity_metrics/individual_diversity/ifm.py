import numpy as np
from msa.tools.optimization_data import SingleRunData, IndivDiversityMetric

__all__ = ["IFM"]


class IFM(IndivDiversityMetric):
    r"""Individual Fitness Mean."""

    def __init__(self, *args, **kwargs):
        r"""Initialize Individual Fitness Mean."""
        super().__init__(*args, **kwargs)

    def _evaluate(self, srd: SingleRunData, *args, **kwargs):
        if srd.populations[0] is None or srd.populations[0].population is None:
            return []
        sums = np.zeros(len(srd.populations[0].population))
        for t in range(len(srd.populations) - 1):
            sums = np.add(sums, srd.populations[t].get_population_fitness_or_empty())

        return sums / len(srd.populations)
