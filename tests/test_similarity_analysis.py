from unittest import TestCase
import numpy as np
import shutil
import os
from msa.diversity_metrics.population_diversity.dpc import DPC
from msa.diversity_metrics.population_diversity.fdc import FDC
from msa.diversity_metrics.population_diversity.pfsd import PFSD
from msa.diversity_metrics.population_diversity.pfm import PFM
from msa.diversity_metrics.individual_diversity.idt import IDT
from msa.diversity_metrics.individual_diversity.isi import ISI
from msa.diversity_metrics.individual_diversity.ifm import IFM
from msa.diversity_metrics.individual_diversity.ifiqr import IFIQR
from msa.tools.optimization_tools import get_sorted_list_of_runs
from niapy.problems.schwefel import Schwefel
from msa.tools.meta_ga import MetaGA, MetaGAFitnessFunction
from msa.tools.metaheuristics_similarity_analyzer import MetaheuristicsSimilarityAnalyzer

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


class TestTargetSimilarity(TestCase):
    def setUp(self):
        self.tmp_path = "./tests/archive_similarity"

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            shutil.rmtree(self.tmp_path)

    def test_target_similarity_analysis(self):
        # Arrange
        pkl_filename = "msa_obj"
        num_runs = 100

        meta_ga = MetaGA(
            fitness_function_type=MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY,
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
            num_runs=num_runs,
            problem=OPTIMIZATION_PROBLEM,
            pop_diversity_metrics=POP_DIVERSITY_METRICS,
            indiv_diversity_metrics=INDIV_DIVERSITY_METRICS,
        )

        analyzer = MetaheuristicsSimilarityAnalyzer(
            meta_ga=meta_ga,
            target_gene_space=TARGET_GENE_SPACES,
            base_archive_path=self.tmp_path,
        )

        # Act
        analyzer.run_similarity_analysis(
            num_comparisons=1,
            get_info=False,
            generate_dataset=True,
            export=True,
            calculate_similarity_metrics=True,
            pkl_filename=pkl_filename,
        )
        analyzer.export_results_to_latex(generate_pdf=True)
        archive_path = analyzer.archive_path
        imported_analyzer = MetaheuristicsSimilarityAnalyzer.import_from_pkl(f"{archive_path}/{pkl_filename}")

        # Assert
        dataset_path = os.path.join(imported_analyzer.dataset_path, "0_comparison")
        first_runs = get_sorted_list_of_runs(dataset_path, imported_analyzer.target_alg_abbr)
        second_runs = get_sorted_list_of_runs(dataset_path, imported_analyzer.optimized_alg_abbr)
        self.assertEquals(len(first_runs), num_runs)
        self.assertEquals(len(second_runs), num_runs)
        self.assertTrue(any(fname.endswith(".pdf") for fname in os.listdir(archive_path)))
        self.assertTrue(any(fname.endswith(".tex") for fname in os.listdir(archive_path)))
        self.assertTrue(any(fname.endswith(".pkl") for fname in os.listdir(archive_path)))

    def test_target_similarity_analysis_target_solutions(self):
        # Arrange
        pkl_filename = "msa_obj"
        num_runs = 100

        meta_ga = MetaGA(
            fitness_function_type=MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY,
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
            num_runs=num_runs,
            problem=Schwefel(10),
            pop_diversity_metrics=POP_DIVERSITY_METRICS,
            indiv_diversity_metrics=INDIV_DIVERSITY_METRICS,
        )

        target_solutions = [
            np.array([1.9, 0.98, 0.83]),
        ]
        analyzer = MetaheuristicsSimilarityAnalyzer(
            meta_ga=meta_ga,
            target_gene_space=TARGET_GENE_SPACES,
            base_archive_path=self.tmp_path,
        )

        # Act
        analyzer.run_similarity_analysis(
            target_solutions=target_solutions,
            get_info=False,
            generate_dataset=True,
            export=True,
            calculate_similarity_metrics=True,
            pkl_filename=pkl_filename,
        )
        analyzer.export_results_to_latex(generate_pdf=True)
        archive_path = analyzer.archive_path
        imported_analyzer = MetaheuristicsSimilarityAnalyzer.import_from_pkl(f"{archive_path}/{pkl_filename}")

        # Assert
        dataset_path = os.path.join(imported_analyzer.dataset_path, "0_comparison")
        first_runs = get_sorted_list_of_runs(dataset_path, imported_analyzer.target_alg_abbr)
        second_runs = get_sorted_list_of_runs(dataset_path, imported_analyzer.optimized_alg_abbr)
        self.assertEquals(len(first_runs), num_runs)
        self.assertEquals(len(second_runs), num_runs)
        self.assertTrue(any(fname.endswith(".pdf") for fname in os.listdir(archive_path)))
        self.assertTrue(any(fname.endswith(".tex") for fname in os.listdir(archive_path)))
        self.assertTrue(any(fname.endswith(".pkl") for fname in os.listdir(archive_path)))
