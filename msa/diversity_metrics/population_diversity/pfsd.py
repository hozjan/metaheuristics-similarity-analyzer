from msa.tools.optimization_data import PopulationData, PopDiversityMetric

__all__ = ['PFSD']


class PFSD(PopDiversityMetric):
    r"""Population Fitness Standard Deviation."""

    def __init__(self, *args, **kwargs):
        r"""Initialize Population Fitness Standard Deviation."""
        super().__init__(*args, **kwargs)

    def _evaluate(self, popData: PopulationData, *args, **kwargs):
        if popData.population_fitness is None:
            return 0
        return popData.population_fitness.std()
