from datetime import datetime
from pathlib import Path
import os
from typing import Any
from tools.meta_ga import MetaGA, MetaGAFitnessFunction
from tools.optimization_tools import optimization_runner
from util.optimization_data import SingleRunData
import numpy as np
import random
from scipy import spatial
import cloudpickle
from niapy.util.factory import (
    _algorithm_options,
    get_algorithm,
)


class MetaheuristicSimilarityAnalyzer:
    r"""Class for search and analysis of similarity of metaheuristic with different parameter settings.
    Uses target metaheuristic with stochastically selected parameters and aims to find parameters of the
    optimized metaheuristic with which they perform in a similar maner."""

    def __init__(
        self,
        meta_ga: MetaGA = None,
        target_gene_space: dict[str, dict[str, Any]] = None,
        comparisons: int = 0,
    ) -> None:
        r"""Initialize the metaheuristic similarity analyzer.

        Args:
            meta_ga (Optional[MetaGA]): Preconfigured instance of the meta genetic algorithm with fitness function set to `TARGET_PERFORMANCE_SIMILARITY`.
            target_gene_space (Optional[Dict[str, Dict[str, Any]]]): Gene space of the reference metaheuristic.
            comparisons (Optional[int]): Number of metaheuristic parameter combinations to analyze during the similarity analysis.
        """
        if (
            not (meta_ga is None)
            and meta_ga.fitness_function_type
            != MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY
        ):
            raise ValueError(
                "Fitness function of the `meta_ga` must be set to `TARGET_PERFORMANCE_SIMILARITY`."
            )

        self.meta_ga = meta_ga
        self.target_gene_space = target_gene_space
        self.comparisons = comparisons
        self.target_solutions = []
        self.optimized_solutions = []
        self.target_feature_vectors = []
        self.optimized_solutions_feature_vectors = []
        self.similarity = []
        self.archive_path = ""
        self.dataset_path = ""
        self.algorithms = []

    def __generate_targets(self):
        low_ranges = []
        high_ranges = []
        for alg_name in self.target_gene_space:
            if alg_name not in _algorithm_options():
                raise KeyError(
                    f"Could not find algorithm by name `{alg_name}` in the niapy library."
                )
            algorithm = get_algorithm(alg_name)
            self.algorithms.append(algorithm.Name[1])
            for setting in self.target_gene_space[alg_name]:
                if type(setting) is tuple:
                    for sub_setting in setting:
                        if type(sub_setting) is tuple:
                            if type(sub_setting[0]) not in [int, float]:
                                raise NameError(
                                    f"Multiplier of the {sub_setting[1]} must be a float or int, {type(sub_setting[0])} found."
                                )
                            if not hasattr(algorithm, sub_setting[1]):
                                raise NameError(
                                    f"Algorithm `{alg_name}` has no attribute named `{sub_setting[1]}`."
                                )
                        else:
                            if not hasattr(algorithm, sub_setting):
                                raise NameError(
                                    f"Algorithm `{alg_name}` has no attribute named `{sub_setting}`."
                                )
                else:
                    if not hasattr(algorithm, setting):
                        raise NameError(
                            f"Algorithm `{alg_name}` has no attribute named `{setting}`."
                        )
                low_ranges.append(self.target_gene_space[alg_name][setting]["low"])
                high_ranges.append(self.target_gene_space[alg_name][setting]["high"])

        for _ in range(self.comparisons):
            target_solution = []
            for low, high in zip(low_ranges, high_ranges):
                target_solution.append(random.uniform(low, high))
            self.target_solutions.append(np.array(target_solution))

    def generate_dataset_from_solutions(self, num_runs: int = None):
        r"""Generate dataset from target and optimized solutions.

        Args:
            num_runs (Optional[int]): Number of runs performed by the metaheuristic for each solution. if None value assigned to meta genetic algorithm is used.
        """
        if num_runs is None:
            num_runs = self.meta_ga.num_runs
        self.similarity = []
        self.target_feature_vectors = []
        self.optimized_solutions_feature_vectors = []
        for idx, (solution_0, solution_1) in enumerate(
            zip(self.target_solutions, self.optimized_solutions)
        ):
            _subset_path = os.path.join(self.dataset_path, f"{idx}_subset")
            if os.path.exists(_subset_path) == False:
                Path(_subset_path).mkdir(parents=True, exist_ok=True)

            solution = np.append(solution_0, solution_1)
            gene_spaces = self.target_gene_space | self.meta_ga.gene_spaces
            algorithms = MetaGA.solution_to_algorithm_attributes(
                solution=solution.tolist(),
                gene_spaces=gene_spaces,
                pop_size=self.meta_ga.pop_size,
            )
            for algorithm in algorithms:
                optimization_runner(
                    algorithm=algorithm,
                    problem=self.meta_ga.problem,
                    runs=num_runs,
                    dataset_path=_subset_path,
                    pop_diversity_metrics=self.meta_ga.pop_diversity_metrics,
                    indiv_diversity_metrics=self.meta_ga.indiv_diversity_metrics,
                    max_iters=self.meta_ga.max_iters,
                    max_evals=self.meta_ga.max_evals,
                    run_index_seed=True,
                    keep_pop_data=False,
                    parallel_processing=True,
                )

            feature_vectors_1 = []
            feature_vectors_2 = []
            for idx, algorithm in enumerate(os.listdir(_subset_path)):
                for problem in os.listdir(os.path.join(_subset_path, algorithm)):
                    runs = os.listdir(os.path.join(_subset_path, algorithm, problem))
                    runs.sort()
                    for run in runs:
                        run_path = os.path.join(_subset_path, algorithm, problem, run)
                        srd = SingleRunData.import_from_json(run_path)
                        feature_vector = srd.get_combined_feature_vector()

                        if idx == 0:
                            feature_vectors_1.append(feature_vector)
                        else:
                            feature_vectors_2.append(feature_vector)

            self.target_feature_vectors.append(feature_vectors_1)
            self.optimized_solutions_feature_vectors.append(feature_vectors_2)
            
            fv1_mean = np.mean(feature_vectors_1, axis=0)
            fv2_mean = np.mean(feature_vectors_2, axis=0)

            self.similarity.append(1 - spatial.distance.cosine(fv1_mean, fv2_mean))

    def __create_folder_structure(self):
        r"""Create folder structure for metaheuristic similarity analysis."""
        datetime_now = str(datetime.now().strftime("%m-%d_%H.%M.%S"))
        self.archive_path = os.path.join(
            "archive/target_performance_similarity",
            "_".join([datetime_now, *self.algorithms, self.meta_ga.problem.name()]),
        )
        self.dataset_path = os.path.join(self.archive_path, "dataset")
        if os.path.exists(self.archive_path) == False:
            Path(self.archive_path).mkdir(parents=True, exist_ok=True)

    def run_similarity_analysis(self, generate_dataset=False):
        r"""Run metaheuristic similarity analysis.

        Args:
            generate_dataset (Optional[bool]): Generate dataset from target and optimized solutions after analysis.
        """
        if (
            self.meta_ga is None
            or self.meta_ga.fitness_function_type
            != MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY
        ):
            raise ValueError(
                "The `meta_ga` parameter must be defined and the fitness function must be set to `TARGET_PERFORMANCE_SIMILARITY`."
            )

        self.__generate_targets()
        self.__create_folder_structure()
        self.meta_ga.base_archive_path = self.archive_path

        for idx, target_solution in enumerate(self.target_solutions):
            target_algorithm = MetaGA.solution_to_algorithm_attributes(
                solution=target_solution,
                gene_spaces=self.target_gene_space,
                pop_size=self.meta_ga.pop_size,
            )
            self.meta_ga.run_meta_ga(
                target_algorithm=target_algorithm[0], prefix=str(idx)
            )
            self.optimized_solutions.append(self.meta_ga.meta_ga.best_solutions[-1])

        if generate_dataset:
            self.generate_dataset_from_solutions()

    def export_to_pkl(self, filename):
        """
        Export instance of the metaheuristic similarity analyzer as .pkl.

        Args:
            filename (str): Filename of the output file. File extension .pkl included upon export.
        """
        filename = os.path.join(self.archive_path, filename)
        msa = cloudpickle.dumps(self)
        with open(filename + ".pkl", "wb") as file:
            file.write(msa)
            cloudpickle.dump(self, file)

    @staticmethod
    def import_from_pkl(filename):
        """
        Import saved instance of the metaheuristic similarity analyzer.

        Args:
            filename (str): Filename of the file to import. File extension .pkl included upon import.

        Returns:
            msa (MetaheuristicSimilarityAnalyzer): Metaheuristic similarity analyzer instance.
        """

        try:
            with open(filename + ".pkl", "rb") as file:
                msa = cloudpickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename}.pkl not found.")
        except:
            raise BaseException(f"File {filename}.pkl could not be loaded.")
        return msa
