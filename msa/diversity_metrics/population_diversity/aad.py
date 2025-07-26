import numpy as np
from msa.tools.optimization_data import PopulationData, PopDiversityMetric
from niapy.util.distances import euclidean

__all__ = ['AAD']


class AAD(PopDiversityMetric):
    r"""Average of the Average Distance around all Particles in the Swarm.

    Reference paper:
        O. Olorunda and A. P. Engelbrecht, "Measuring exploration/exploitation in particle swarms using swarm
        diversity," China, 2008, pp. 1128-1134, doi: 10.1109/CEC.2008.4630938.

    """

    def __init__(self, *args, **kwargs):
        r"""Initialize Average of the Average Distance around all Particles in the Swarm."""
        super().__init__(*args, **kwargs)

    def _evaluate(self, popData: PopulationData, *args, **kwargs):
        if popData.population is None:
            return 0
        P, N = np.shape(popData.population)
        aad = 0.0

        for pi in popData.population:
            ad = 0.0
            for p in popData.population:
                ad += euclidean(p, pi)
            aad += ad / P

        return aad / P
