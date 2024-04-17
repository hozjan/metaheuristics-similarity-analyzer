RNG_SEED = 42
DATASET_PATH = "./meta_dataset"
POP_SIZE = 40
BATCH_SIZE = 10
EPOCHS = 10
MAX_ITER = 20
NUM_RUNS = 10
META_GA_GENERATIONS = 3
META_GA_SOLUTIONS_PER_POP = 10

# problem, algorithm and parameter names must match those from the niapy library
OPTIMIZATION_PROBLEM = "Schwefel"
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
