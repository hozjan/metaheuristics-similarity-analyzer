from util.indiv_diversity_metrics import IndivDiversityMetric
from util.pop_diversity_metrics import PopDiversityMetric
from tools.problems.schwefel import Schwefel
from tools.algorithms.fa import FireflyAlgorithm
from tools.algorithms.pso import ParticleSwarmAlgorithm
import numpy as np


RNG_SEED = 42

"""
Optimization parameters
"""
DATASET_PATH = "./dataset"
POP_SIZE = 30
MAX_EVALS = 10000  # per run
NUM_RUNS = 30  # per algorithm
# metrics to calculate when performing optimization
POP_DIVERSITY_METRICS = [
    PopDiversityMetric.PDC,
    PopDiversityMetric.FDC,
    PopDiversityMetric.PFSD,
    PopDiversityMetric.PFM,
]
INDIV_DIVERSITY_METRICS = [
    IndivDiversityMetric.IDT,
    IndivDiversityMetric.ISI,
    IndivDiversityMetric.IFM,
    IndivDiversityMetric.IFIQR,
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
NUM_COMPARISONS = 30

"""
Meta GA parameters
"""
META_GA_GENERATIONS = 40
META_GA_SOLUTIONS_PER_POP = 30
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
"""
GENE_SPACES = {
    FireflyAlgorithm: {
        "alpha": {"low": 0.01, "high": 1.0, "step": 0.01},
        "beta0": {"low": 0.01, "high": 1.0, "step": 0.01},
        "gamma": {"low": 0.0, "high": 1.0, "step": 0.001},
        "theta": {"low": 0.95, "high": 1.0, "step": 0.001},
    },
}
"""
GENE_SPACES = {
    "BatAlgorithm": {
        "loudness": {"low": 0.01, "high": 1.0, "step": 0.01},
        "pulse_rate": {"low": 0.01, "high": 1.0, "step": 0.01},
        "alpha": {"low": 0.9, "high": 1.0, "step": 0.001},
        "gamma": {"low": 0.0, "high": 1.0, "step": 0.01},
    }
}
"""
GENE_SPACES = {
    ParticleSwarmAlgorithm: {
        "c1": {"low": 0.01, "high": 2.5, "step": 0.01},
        "c2": {"low": 0.01, "high": 2.5, "step": 0.01},
        "w": {"low": 0.0, "high": 1.0, "step": 0.01},
    },
}
"""

TARGET_GENE_SPACES = {
    "ParticleSwarmAlgorithm": {
        "c1": {"low": 0.01, "high": 2.5, "step": 0.01},
        "c2": {"low": 0.01, "high": 2.5, "step": 0.01},
        "w": {"low": 0.0, "high": 1.0, "step": 0.01},
    }
}