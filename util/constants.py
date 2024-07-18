from util.indiv_diversity_metrics import IndivDiversityMetric
from util.pop_diversity_metrics import PopDiversityMetric
from niapy.problems.sphere import Sphere
from niapy.problems.rosenbrock import Rosenbrock
from niapy.problems.rastrigin import Rastrigin
from niapy.problems.schwefel import Schwefel
from niapy.problems.bent_cigar import BentCigar
from niapy.problems.trid import Trid
from niapy.problems.zakharov import Zakharov
from niapy.problems.alpine import Alpine1
from niapy.problems.griewank import Griewank
import numpy as np


RNG_SEED = 42

"""
Optimization parameters
"""
DATASET_PATH = "./dataset"
POP_SIZE = 30
MAX_EVALS = 15000  # per run
MAX_ITERS = 500  # per run
NUM_RUNS = 50  # per algorithm
# metrics to calculate when performing optimization
POP_DIVERSITY_METRICS = [
    PopDiversityMetric.PDC,
    PopDiversityMetric.FDC,
    PopDiversityMetric.PFSD,
    PopDiversityMetric.PFMea,
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
BATCH_SIZE = 20
EPOCHS = 100
N_PCA_COMPONENTS = 3
LSTM_NUM_LAYERS = 3
LSTM_HIDDEN_DIM = 128
LSTM_DROPOUT = 0.2
VAL_SIZE = 0.2
TEST_SIZE = 0.2


"""
Metaheuristic similarity analyzer parameters
"""
NUM_COMPARISONS = 1


"""
Meta GA parameters
"""
META_GA_GENERATIONS = 20
META_GA_SOLUTIONS_PER_POP = 20
META_GA_PERCENT_PARENTS_MATING = 60
META_GA_PARENT_SELECTION_TYPE = "tournament"
META_GA_K_TOURNAMENT = 2
META_GA_CROSSOVER_PROBABILITY = 0.9
META_GA_CROSSOVER_TYPE = "uniform"
META_GA_MUTATION_NUM_GENES = 1
META_GA_MUTATION_TYPE = "random"
META_GA_KEEP_ELITISM = 1

# problem, algorithm and parameter names must match those from the niapy library
OPTIMIZATION_PROBLEM = Schwefel(dimension=20)
GENE_SPACES = {
    "FireflyAlgorithm": {
        "alpha": {"low": 0.1, "high": 1.0},
        "beta0": {"low": 0.0, "high": 1.0},
        "gamma": {"low": 0.0, "high": 0.01},
        "theta": {"low": 0.97, "high": 1.0},
    },
}

TARGET_GENE_SPACES = {
    "ParticleSwarmAlgorithm": {
        "c1": {"low": 0.0, "high": 2.5},
        "c2": {"low": 0.0, "high": 2.5},
        "w": {"low": 0.0, "high": 1.1},
    }
}
