from util.indiv_diversity_metrics import IndivDiversityMetric
from util.pop_diversity_metrics import PopDiversityMetric

RNG_SEED = 42

"""
Optimization parameters
"""
DATASET_PATH = "./dataset"
POP_SIZE = 40
MAX_EVALS = 10000 # per run
NUM_RUNS = 300 # per algorithm
# metrics to calculate when performing optimization
POP_DIVERSITY_METRICS = [
    PopDiversityMetric.PDC,
    PopDiversityMetric.PED,
    PopDiversityMetric.PMD,
    PopDiversityMetric.AAD,
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


"""
Meta GA parameters
"""
META_GA_GENERATIONS = 20
META_GA_SOLUTIONS_PER_POP = 10
# problem, algorithm and parameter names must match those from the niapy library
OPTIMIZATION_PROBLEM = "Ackley"
GENE_SPACES = {
    "FireflyAlgorithm": {
        "alpha": {"low": 0.01, "high": 1},
        "beta0": {"low": 0.01, "high": 10},
        "gamma": {"low": 0.01, "high": 2},
        "theta": {"low": 0.01, "high": 2},
    },
    "BatAlgorithm": {
        "loudness": {"low": 0.01, "high": 1},
        "pulse_rate": {"low": 0.01, "high": 1},
        "alpha": {"low": 0.01, "high": 1},
        "gamma": {"low": 0.01, "high": 1},
    },
    "ParticleSwarmOptimization": {
        "c1": {"low": 0.01, "high": 1},
        "c2": {"low": 0.01, "high": 1},
        "w": {"low": 0.01, "high": 1},
    },
}
