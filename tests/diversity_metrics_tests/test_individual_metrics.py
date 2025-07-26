from unittest import TestCase
import numpy as np
from tests.diversity_metrics_tests.util.optimization_data import GenerateSingleRunData
from msa.diversity_metrics.individual_diversity.idt import IDT
from msa.diversity_metrics.individual_diversity.ifiqr import IFIQR
from msa.diversity_metrics.individual_diversity.ifm import IFM
from msa.diversity_metrics.individual_diversity.isi import ISI
from niapy.problems.sphere import Sphere

DIMENSION = 2
POP_SIZE = 2
ITERATIONS = 100


class TestIndivDiversityMetrics(TestCase):
    def setUp(self):
        problem = Sphere(DIMENSION)
        self.problem = problem
        self.srd = GenerateSingleRunData(POP_SIZE, DIMENSION, ITERATIONS, problem)

    def test_IDT(self):
        metric = IDT()
        diversity = metric.evaluate(self.srd)
        np.testing.assert_array_equal([0.0, 0.0], diversity)

    def test_IFIQR(self):
        metric = IFIQR()
        diversity = metric.evaluate(self.srd)
        np.testing.assert_array_equal([0.0, 0.0], diversity)

    def test_IFM(self):
        metric = IFM()
        diversity = metric.evaluate(self.srd)
        np.testing.assert_array_equal([0.0, 0.0], diversity)

    def test_ISI(self):
        metric = ISI()
        diversity = metric.evaluate(self.srd)
        np.testing.assert_array_equal([0.0, 0.0], diversity)
