from msa.tools.optimization_data import PopulationData, PopDiversityMetric
from niapy.util.distances import euclidean

__all__ = ['PED']


class PED(PopDiversityMetric):
    r"""Population Euclidean Distance."""

    def __init__(self, *args, **kwargs):
        """Initialize Population Euclidean Distance."""
        super().__init__(*args, **kwargs)

    def _evaluate(self, popData: PopulationData, *args, **kwargs):
        if popData.population is None:
            return 0
        ped = 0

        for index_i, pi in enumerate(popData.population):
            for pj in popData.population[index_i + 1:]:
                ped += euclidean(pi, pj)
        return ped
