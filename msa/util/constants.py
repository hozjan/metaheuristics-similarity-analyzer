from msa.diversity_metrics.population_diversity.dpc import DPC
from msa.diversity_metrics.population_diversity.fdc import FDC
from msa.diversity_metrics.population_diversity.pfsd import PFSD
from msa.diversity_metrics.population_diversity.pfm import PFM
from msa.diversity_metrics.individual_diversity.idt import IDT
from msa.diversity_metrics.individual_diversity.isi import ISI
from msa.diversity_metrics.individual_diversity.ifm import IFM
from msa.diversity_metrics.individual_diversity.ifiqr import IFIQR
from niapy.problems.schwefel import Schwefel

RNG_SEED = 42

# problem, algorithm and parameter names must match those from the niapy framework
OPTIMIZATION_PROBLEM = Schwefel(dimension=20)

"""
Optimization parameters
"""
DATASET_PATH = "./dataset"
POP_SIZE = 30
MAX_EVALS = 10000  # per run
NUM_RUNS = 30  # per algorithm

# metrics to calculate when performing optimization
POP_DIVERSITY_METRICS = [
    DPC(OPTIMIZATION_PROBLEM),
    FDC(OPTIMIZATION_PROBLEM, [420.968746], True),
    PFSD(),
    PFM(),
]
INDIV_DIVERSITY_METRICS = [
    IDT(),
    ISI(),
    IFM(),
    IFIQR(),
]

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
