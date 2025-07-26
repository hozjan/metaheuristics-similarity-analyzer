from msa.tools.optimization_data import PopulationData, PopDiversityMetric

__all__ = ['PMD']


class PMD(PopDiversityMetric):
    r"""Population Manhattan Distance."""

    def __init__(self, *args, **kwargs):
        """Initialize Population Manhattan Distance."""
        super().__init__(*args, **kwargs)

    def _evaluate(self, popData: PopulationData, *args, **kwargs):
        if popData.population is None:
            return 0
        pmd = 0

        for index_i, pi in enumerate(popData.population):
            for pj in popData.population[index_i + 1:]:
                sum = 0
                for xi, xj in zip(pi, pj):
                    sum += abs(xi - xj)
                pmd += sum

        return pmd
