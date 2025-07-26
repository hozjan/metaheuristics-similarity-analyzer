from unittest import TestCase
from tests.diversity_metrics_tests.util.optimization_data import GeneratePopulationData
from msa.diversity_metrics.population_diversity.dpc import DPC
from msa.diversity_metrics.population_diversity.fdc import FDC
from msa.diversity_metrics.population_diversity.aad import AAD
from msa.diversity_metrics.population_diversity.pdi import PDI
from msa.diversity_metrics.population_diversity.ped import PED
from msa.diversity_metrics.population_diversity.pfm import PFM
from msa.diversity_metrics.population_diversity.pfsd import PFSD
from msa.diversity_metrics.population_diversity.pmd import PMD
from niapy.problems.sphere import Sphere

DIMENSION = 5
POP_SIZE = 10


class TestPopDiversityMetrics(TestCase):
    def setUp(self):
        problem = Sphere(DIMENSION)
        self.problem = problem
        self.popData = GeneratePopulationData(POP_SIZE, DIMENSION, problem)

    def test_DPC(self):
        metric = DPC(self.problem)
        diversity = metric.evaluate(self.popData)
        self.assertEquals(0.0, diversity)

    def test_FDC(self):
        metric = FDC(self.problem, [0.0], True)
        diversity = metric.evaluate(self.popData)
        self.assertEquals(1.0, diversity)

    def test_AAD(self):
        metric = AAD()
        diversity = metric.evaluate(self.popData)
        self.assertEquals(0.0, diversity)

    def test_PDI(self):
        metric = PDI(self.problem)
        diversity = metric.evaluate(self.popData)
        self.assertAlmostEqual(0.0, diversity, places=10)

    def test_PED(self):
        metric = PED(self.problem)
        diversity = metric.evaluate(self.popData)
        self.assertEquals(0.0, diversity)

    def test_PFM(self):
        metric = PFM(self.problem)
        diversity = metric.evaluate(self.popData)
        self.assertEquals(0.0, diversity)

    def test_PFSD(self):
        metric = PFSD(self.problem)
        diversity = metric.evaluate(self.popData)
        self.assertEquals(0.0, diversity)

    def test_PMD(self):
        metric = PMD(self.problem)
        diversity = metric.evaluate(self.popData)
        self.assertEquals(0.0, diversity)
