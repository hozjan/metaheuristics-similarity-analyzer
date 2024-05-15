from util.indiv_diversity_metrics import IndivDiversityMetric
from util.pop_diversity_metrics import PopDiversityMetric
from niapy.problems.sphere import Sphere
from niapy.problems.rosenbrock import Rosenbrock
from niapy.problems.bent_cigar import BentCigar
from niapy.problems.trid import Trid
from niapy.problems.zakharov import Zakharov
import numpy as np


RNG_SEED = 42

"""
Optimization parameters
"""
DATASET_PATH = "./dataset"
POP_SIZE = 30
MAX_EVALS = 10000  # per run
MAX_ITERS = 150  # per run
NUM_RUNS = 100  # per algorithm
# metrics to calculate when performing optimization
POP_DIVERSITY_METRICS = [
    PopDiversityMetric.AAD,
    PopDiversityMetric.PDC,
    PopDiversityMetric.PFSD,
    PopDiversityMetric.PFMea,
    PopDiversityMetric.PFMed,
]
INDIV_DIVERSITY_METRICS = [
    IndivDiversityMetric.IDT,
    IndivDiversityMetric.ISI,
    IndivDiversityMetric.IFMea,
    IndivDiversityMetric.IFMed,
]


"""
Machine learning parameters
"""
BATCH_SIZE = 10
EPOCHS = 100
LSTM_NUM_LAYERS = 3
LSTM_HIDDEN_DIM = 256
LSTM_DROPOUT = 0.3
VAL_SIZE = 0.2
TEST_SIZE = 0.2


"""
Meta GA parameters
"""
META_GA_GENERATIONS = 10
META_GA_SOLUTIONS_PER_POP = 10

# problem, algorithm and parameter names must match those from the niapy library
OPTIMIZATION_PROBLEM = Rosenbrock(dimension=20)
GENE_SPACES = {
    "FireflyAlgorithm": {
        "alpha": {"low": 0.0, "high": 1.1},
        "beta0": {"low": 0.0, "high": 1.1},
        "gamma": {"low": 0.0, "high": 1.1},
        "theta": {"low": 0.0, "high": 1.1},
    },
    "ParticleSwarmAlgorithm": {
        "c1": {"low": 0.0, "high": 3.0},
        "c2": {"low": 0.0, "high": 3.0},
        "w": {"low": 0.0, "high": 2.0},
        "min_velocity": {"low": -10, "high": 0.0},
        "max_velocity": {"low": 0.0, "high": 10.0},
    },
}
