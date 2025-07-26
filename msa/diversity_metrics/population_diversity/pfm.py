from msa.tools.optimization_data import PopulationData, PopDiversityMetric

__all__ = ['PFM']


class PFM(PopDiversityMetric):
    r"""Population Fitness Mean."""

    def __init__(self, *args, **kwargs):
        r"""Initialize Population Fitness Mean."""
        super().__init__(*args, **kwargs)

    def _evaluate(self, popData: PopulationData, *args, **kwargs):
        if popData.population_fitness is None:
            return 0
        return popData.population_fitness.mean()
