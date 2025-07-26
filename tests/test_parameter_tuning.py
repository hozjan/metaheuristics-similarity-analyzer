from unittest import TestCase
import os.path
import pygad
from niapy.problems.schwefel import Schwefel
from msa.tools.meta_ga import MetaGA, MetaGAFitnessFunction


GENE_SPACES = {
    "BatAlgorithm": {
        "loudness": {"low": 0.01, "high": 1.0, "step": 0.01},
        "pulse_rate": {"low": 0.01, "high": 1.0, "step": 0.01},
        "alpha": {"low": 0.9, "high": 1.0, "step": 0.001},
        "gamma": {"low": 0.0, "high": 1.0, "step": 0.01},
    }
}

TARGET_GENE_SPACES = {
    "ParticleSwarmAlgorithm": {
        "c1": {"low": 0.01, "high": 2.5, "step": 0.01},
        "c2": {"low": 0.01, "high": 2.5, "step": 0.01},
        "w": {"low": 0.0, "high": 1.0, "step": 0.01},
    }
}

OPTIMIZATION_PROBLEM = Schwefel(10)


class TestParameterTuning(TestCase):
    def test_parameter_tuning(self):
        # Arrange
        meta_ga = MetaGA(
            fitness_function_type=MetaGAFitnessFunction.PARAMETER_TUNING,
            ga_generations=3,
            ga_solutions_per_pop=5,
            ga_percent_parents_mating=60,
            ga_parent_selection_type="tournament",
            ga_k_tournament=2,
            ga_crossover_type="uniform",
            ga_mutation_type="random",
            ga_crossover_probability=0.9,
            ga_mutation_num_genes=1,
            ga_keep_elitism=1,
            gene_spaces=GENE_SPACES,
            pop_size=10,
            max_evals=100,
            num_runs=100,
            problem=OPTIMIZATION_PROBLEM,
            base_archive_path="./tests/archive/parameter_tuning",
        )

        # Act
        meta_ga_filename = "meta_ga_obj"
        meta_ga.run_meta_ga(filename=meta_ga_filename, prefix="0", get_info=False)
        meta_ga_save = pygad.load(f"./tests/archive/parameter_tuning/0_BA_Schwefel/{meta_ga_filename}")

        # Assert
        self.assertIsNotNone(meta_ga_save)
        self.assertTrue(os.path.isfile("./tests/archive/parameter_tuning/0_BA_Schwefel/meta_ga_log_file.txt"))
        self.assertTrue(os.path.isfile("./tests/archive/parameter_tuning/0_BA_Schwefel/meta_ga_fitness_plot.png"))
