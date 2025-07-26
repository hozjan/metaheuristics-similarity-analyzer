import numpy as np
from msa.tools.optimization_data import SingleRunData, IndivDiversityMetric

__all__ = ["IDT"]


class IDT(IndivDiversityMetric):
    r"""Individual Distance Traveled."""

    def __init__(self, *args, **kwargs):
        r"""Initialize Individual Distance Traveled."""
        super().__init__(*args, **kwargs)

    def _evaluate(self, srd: SingleRunData, *args, **kwargs):
        populations = srd.populations
        distances = []

        for t in range(len(populations) - 1):
            first = np.array([pop for pop in populations[t].get_population_or_empty()])
            second = np.array([pop for pop in populations[t + 1].get_population_or_empty()])
            distances.append(np.linalg.norm(first - second, axis=1))

        distances = np.sum(distances, axis=0)

        return distances
