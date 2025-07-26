import numpy as np
from niapy.util.distances import euclidean
from msa.tools.optimization_data import SingleRunData, IndivDiversityMetric

__all__ = ["ISI"]


class ISI(IndivDiversityMetric):
    r"""Individual Sinuosity Index."""

    def __init__(self, *args, **kwargs):
        r"""Initialize Individual Sinuosity Index."""
        super().__init__(*args, **kwargs)

    def _evaluate(self, srd: SingleRunData, *args, **kwargs):
        populations = srd.populations
        distances = []
        for t in range(len(srd.populations) - 1):
            if populations is None or populations[t] is None or populations[t + 1] is None:
                continue
            first = np.array([pop for pop in populations[t].get_population_or_empty()])
            second = np.array([pop for pop in populations[t + 1].get_population_or_empty()])
            distances.append(np.linalg.norm(first - second, axis=1))

        idt = np.sum(distances, axis=0)
        isi: list[float] = []
        for p in range(len(populations[0].get_population_or_empty())):
            # calculate euclidean distance between positions in first and last iteration
            d = euclidean(
                populations[0].get_population_or_empty()[p],
                populations[len(populations) - 1].get_population_or_empty()[p],
            )
            if d != 0:
                isi.append(idt[p] / d)
            else:
                isi.append(0.0)

        return isi
