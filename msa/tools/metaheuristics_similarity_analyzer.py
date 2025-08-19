from datetime import datetime
from pathlib import Path
import warnings
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
)
from sklearn import svm
from matplotlib import pyplot as plt
from pylatex import Document, Section, Subsection
from pylatex import MultiColumn, Package, LongTable
from pylatex.utils import bold, NoEscape
from msa.util.helper import random_float_with_step, get_algorithm_by_name, timer
from msa.tools.meta_ga import MetaGA, MetaGAFitnessFunction
from msa.tools.optimization_tools import optimization_runner, get_sorted_list_of_runs
from msa.tools.optimization_data import SingleRunData
import numpy as np
import numpy.typing as npt
from scipy import spatial, stats
import graphviz
import cloudpickle
from niapy.algorithms import Algorithm
from enum import Enum
import pandas as pd

__all__ = ["MetaheuristicsSimilarityAnalyzer", "SimilarityMetrics"]


class SimilarityMetrics(Enum):
    SMAPE = "smape"
    COS = "cos"
    SPEARMAN = "spearman"
    KNN = "knn"
    SVM = "svm"


class MetaheuristicsSimilarityAnalyzer:
    r"""Class for the analysis of the similarity of metaheuristics with
    different parameter settings. Uses target/reference metaheuristic with
    either randomly or by parameter tuning selected parameters and aims to
    find parameters of the optimized metaheuristic with which they perform
    in a similar manner.

    Attributes:
        meta_ga (MetaGA): Pre-configured instance of the MetaGA with fitness function set to
            `TARGET_PERFORMANCE_SIMILARITY`.
        target_gene_space (dict[str | Algorithm, dict[str, dict[str, float]]]):
            Gene space of the target/reference metaheuristic.
        base_archive_path (str): Base archive path of the MSA. Used for dataset location.
        archive_path (str): Archive path of the MSA including `base_archive_path` followed by /`prefix`_.
            Generated when calling `run_similarity_analysis`.
        dataset_path (str): Path of the generated dataset including `archive_path`.
        target_solutions (list[np.ndarray]): List of target solutions used for the target (reference)
            algorithm during the analysis
        optimized_solutions (list[np.ndarray]): List of optimized solutions of the optimized algorithm
            acquired during the similarity analysis.
        similarity_metrics (dict[str, list[float]]): Dictionary of values of the calculated similarity
            metrics.
    """

    def __init__(
        self,
        meta_ga: MetaGA,
        target_gene_space: dict[str | Algorithm, dict[str, dict[str, float]]],
        base_archive_path: str = "archive",
    ) -> None:
        r"""Initialize the metaheuristic similarity analyzer.

        Args:
            meta_ga (MetaGA): Pre-configured instance of the MetaGA with fitness function set to
                `TARGET_PERFORMANCE_SIMILARITY`.
            target_gene_space (dict[str | Algorithm, dict[str, dict[str, float]]]):
                Gene space of the target/reference metaheuristic.
            base_archive_path (Optional[str]): Base archive path of the MSA. Used for dataset location.

        Raises:
            ValueError: Incorrect `fitness_function_type` value assigned to meta_ga.
            ValueError: Incorrect number of gene space in `target_gene_space`.
        """

        if meta_ga is not None and meta_ga.fitness_function_type != MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY:
            raise ValueError(
                """`fitness_function_type` of the `meta_ga` must be set to
                `TARGET_PERFORMANCE_SIMILARITY`."""
            )

        if len(target_gene_space) != 1:
            raise ValueError(
                """Only one algorithm must be defined in `target_gene_space`
                provided."""
            )

        self.meta_ga = meta_ga
        self.target_gene_space = target_gene_space
        self.target_solutions: list[npt.NDArray] = []
        self.optimized_solutions: list[list[float]] = []
        self.similarity_metrics: dict[str, list[float]] = {}
        self.archive_path = ""
        self.dataset_path = ""
        self.base_archive_path = base_archive_path
        self.__target_alg_abbr = get_algorithm_by_name(list(target_gene_space)[0]).Name[1]
        self.__optimized_alg_abbr = get_algorithm_by_name(list(self.meta_ga.gene_space)[0]).Name[1]
        self.__meta_ga_pkl_filename = "meta_ga_export"
        self.__comparison_dir_suffix = "comparison"
        self.__dataset_dirname = "dataset"
        self.__absolute_dirname = None

    def __generate_targets_and_folder_structure(
        self,
        num_comparisons: int,
        generate_optimized_targets: bool = False,
        prefix: str | None = None,
    ):
        r"""Generate target solutions and folder structure.

        Args:
            generate_optimized_targets (Optional[bool]): Generate target
                solutions by parameter tuning if True, otherwise generate
                random targets.
            num_comparisons (Optional[int]): Number of metaheuristic parameter
                combinations to analyze during the similarity analysis.
            prefix (Optional[str]): Use custom prefix for the name of the base
                folder in structure. Uses current datetime by default.

        Raises:
            NameError: Algorithm does not have the attribute provided in the `gene_space`.
        """
        low_ranges = []
        high_ranges = []
        steps = []

        for alg_name in self.target_gene_space:
            algorithm = get_algorithm_by_name(alg_name)
            for setting in self.target_gene_space[alg_name]:
                if not hasattr(algorithm, setting):
                    raise NameError(f"Algorithm `{alg_name}` has no attribute named `{setting}`.")
                low_ranges.append(self.target_gene_space[alg_name][setting]["low"])
                high_ranges.append(self.target_gene_space[alg_name][setting]["high"])
                steps.append(self.target_gene_space[alg_name][setting]["step"])

        self.__create_folder_structure(prefix=prefix)

        for idx in range(num_comparisons):
            if generate_optimized_targets:
                meta_ga = MetaGA(
                    fitness_function_type=MetaGAFitnessFunction.PARAMETER_TUNING,
                    ga_generations=self.meta_ga.ga_generations,
                    ga_solutions_per_pop=self.meta_ga.ga_solutions_per_pop,
                    ga_percent_parents_mating=self.meta_ga.ga_percent_parents_mating,
                    ga_parent_selection_type=self.meta_ga.ga_parent_selection_type,
                    ga_k_tournament=self.meta_ga.ga_k_tournament,
                    ga_crossover_type=self.meta_ga.ga_crossover_type,
                    ga_mutation_type=self.meta_ga.ga_mutation_type,
                    ga_crossover_probability=self.meta_ga.ga_crossover_probability,
                    ga_mutation_num_genes=self.meta_ga.ga_mutation_num_genes,
                    ga_keep_elitism=self.meta_ga.ga_keep_elitism,
                    gene_space=self.target_gene_space,
                    pop_size=self.meta_ga.pop_size,
                    max_evals=self.meta_ga.max_evals,
                    num_runs=self.meta_ga.num_runs,
                    problem=self.meta_ga.problem,
                    base_archive_path=os.path.join(self.archive_path, "target_tuning"),
                )
                target_solution = meta_ga.run_meta_ga(prefix=str(idx))
                if target_solution is not None:
                    self.target_solutions.append(target_solution)
            else:
                target_solution = []
                for low, high, step in zip(low_ranges, high_ranges, steps):
                    target_solution.append(random_float_with_step(low=low, high=high, step=step))
                self.target_solutions.append(np.array(target_solution))

    def calculate_similarity_metrics(self):
        r"""Calculates similarity metrics from diversity metrics
        values of the comparisons stored in the generated dataset.
        If no dataset was created method will have no effect.

        Raises:
            FileNotFoundError: No dataset found.
        """

        if os.path.exists(self.dataset_path) is False:
            raise FileNotFoundError(
                "Dataset does not exist. Run `generate_dataset_from_solutions` to generate dataset!"
            )

        comparisons = os.listdir(self.dataset_path)

        mean_smape = []
        cosine_similarity = []
        spearman_r = []

        for idx in range(len(comparisons)):
            comparison = f"{idx}_comparison"
            feature_vectors_1 = []
            feature_vectors_2 = []

            first_runs = get_sorted_list_of_runs(os.path.join(self.dataset_path, comparison), self.__target_alg_abbr)
            second_runs = get_sorted_list_of_runs(
                os.path.join(self.dataset_path, comparison), self.__optimized_alg_abbr
            )

            smape_values = []
            for first_run, second_run in zip(first_runs, second_runs):
                f_srd = SingleRunData.import_from_json(first_run)
                s_srd = SingleRunData.import_from_json(second_run)

                f_feature_vector = f_srd.get_feature_vector()
                s_feature_vector = s_srd.get_feature_vector()

                feature_vectors_1.append(f_feature_vector)
                feature_vectors_2.append(s_feature_vector)

                # calculate 1-SMAPE metric
                smape_values.append(f_srd.get_diversity_metrics_similarity(s_srd))

            mean_smape.append(round(np.mean(smape_values), 2))

            fv1_mean = np.mean(feature_vectors_1, axis=0)
            fv2_mean = np.mean(feature_vectors_2, axis=0)

            # calculate cosine similarity and spearman correlation coefficients
            cosine_similarity.append(1 - spatial.distance.cosine(fv1_mean, fv2_mean))
            r, p = stats.spearmanr(fv1_mean, fv2_mean)
            spearman_r.append(r)

        # get knn and svm 1-accuracy metric
        ml_accuracy = self.svm_and_knn_classification_similarity_metrics(100)

        self.similarity_metrics[SimilarityMetrics.SMAPE.value] = mean_smape
        self.similarity_metrics[SimilarityMetrics.COS.value] = cosine_similarity
        self.similarity_metrics[SimilarityMetrics.SPEARMAN.value] = spearman_r
        self.similarity_metrics.update(ml_accuracy)

    def generate_dataset_from_solutions(self, num_runs: int | None = None):
        r"""Generate dataset from target and optimized solutions.

        Args:
            num_runs (Optional[int]): Number of runs performed by the
                metaheuristic for each solution. if None the value of `num_runs`
                assigned to the `meta_ga` is used.
        """
        if num_runs is None:
            num_runs = self.meta_ga.num_runs

        for idx, (target_solution, optimized_solution) in enumerate(
            zip(self.target_solutions, self.optimized_solutions)
        ):
            _comparison_path = os.path.join(self.dataset_path, f"{idx}_comparison")
            if os.path.exists(_comparison_path) is False:
                Path(_comparison_path).mkdir(parents=True, exist_ok=True)

            target_algorithm = MetaGA.solution_to_algorithm_attributes(
                solution=target_solution,
                gene_space=self.target_gene_space,
                pop_size=self.meta_ga.pop_size,
            )
            optimized_algorithm = MetaGA.solution_to_algorithm_attributes(
                solution=optimized_solution,
                gene_space=self.meta_ga.gene_space,
                pop_size=self.meta_ga.pop_size,
            )
            for algorithm in (target_algorithm, optimized_algorithm):
                optimization_runner(
                    algorithm=algorithm,
                    problem=self.meta_ga.problem,
                    runs=num_runs,
                    dataset_path=_comparison_path,
                    pop_diversity_metrics=self.meta_ga.pop_diversity_metrics,
                    indiv_diversity_metrics=self.meta_ga.indiv_diversity_metrics,
                    max_evals=self.meta_ga.max_evals,
                    run_index_seed=True,
                    keep_pop_data=False,
                    parallel_processing=True,
                )

    def __create_folder_structure(self, prefix: str | None = None):
        r"""Create folder structure for metaheuristic similarity analysis.

        Args:
            prefix (Optional[str]): Use custom prefix for the name of the base
                folder in structure. Uses current datetime by default.
        """
        if prefix is None:
            prefix = str(datetime.now().strftime("%m-%d_%H.%M.%S"))
        self.archive_path = os.path.join(
            self.base_archive_path,
            "_".join(
                [
                    prefix,
                    f"{self.__target_alg_abbr}-{self.__optimized_alg_abbr}",
                    self.meta_ga.problem.name(),
                ]
            ),
        )
        self.dataset_path = os.path.join(self.archive_path, self.__dataset_dirname)
        if os.path.exists(self.archive_path) is False:
            Path(self.archive_path).mkdir(parents=True, exist_ok=True)

    def run_similarity_analysis(
        self,
        num_comparisons: int | None = None,
        target_solutions: list[npt.NDArray] | None = None,
        generate_optimized_targets: bool = False,
        get_info: bool = False,
        generate_dataset: bool = False,
        num_runs: int | None = None,
        calculate_similarity_metrics: bool = False,
        prefix: str | None = None,
        export: bool = False,
        pkl_filename: str = "msa_obj",
    ):
        r"""Run metaheuristic similarity analysis.

        Args:
            num_comparisons (Optional[int]): Number of metaheuristic parameter
                combinations to analyze during the similarity analysis.
                Required if `target_solutions` is None.
            target_solutions (Optional[list[numpy.ndarray]]): Target solutions
                used for the target algorithm. Generated if None.
            generate_optimized_targets (Optional[bool]): Generate target
                solutions by parameter tuning. Target solutions wil be
                generated by uniform rng if false. Has no effect if
                `target_solutions` is not None.
            get_info (Optional[bool]): Generate info scheme of the
                metaheuristic similarity analyzer (false by default).
            generate_dataset (Optional[bool]): Generate dataset from
                target and optimized solutions after analysis
                (false by default).
            num_runs (Optional[int]): Number of runs performed by the
                metaheuristic for each solution when generating the dataset.
                if None the value of `num_runs` assigned to the `meta_ga` is used.
            calculate_similarity_metrics (Optional[bool]): Calculates
                similarity metrics from target and optimized solutions
                after analysis (false by default). Has no effect if
                `generate_dataset` is false.
            prefix (Optional[str]): Use custom prefix for the name of the base
                folder in structure. Uses current datetime by default.
            export (Optional[bool]): Export MSA object to pkl after analysis.
            pkl_filename (Optional[str]): Filename of the exported .pkl file.
                Used if `export` is true.

        Raises:
            ValueError: `meta_ga` not defined or `fitness_function_type`
                has incorrect value.
        """
        if (
            self.meta_ga is None
            or self.meta_ga.fitness_function_type != MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY
        ):
            raise ValueError(
                """The `meta_ga` parameter must be defined and the fitness
                function must be set to `TARGET_PERFORMANCE_SIMILARITY`."""
            )

        if target_solutions is None and num_comparisons is None:
            raise ValueError("""None of the `num_comparisons` or `target_solutions` was defined!""")

        if target_solutions is None and num_comparisons is not None:
            self.__generate_targets_and_folder_structure(num_comparisons, generate_optimized_targets, prefix)
        elif target_solutions is not None:
            self.target_solutions = target_solutions
            self.__create_folder_structure(prefix=prefix)

        self.meta_ga.base_archive_path = self.archive_path

        if get_info:
            self.msa_info(
                filename=os.path.join(self.archive_path, "msa_info"),
            )

        start = time.time()
        for comparison_idx, target_solution in enumerate(self.target_solutions):
            target_algorithm = MetaGA.solution_to_algorithm_attributes(
                solution=target_solution,
                gene_space=self.target_gene_space,
                pop_size=self.meta_ga.pop_size,
            )
            logger_headline = f"\n======> {comparison_idx}/{len(self.target_solutions)-1}"
            logger_headline += f"_COMPARISON_{self.__target_alg_abbr}-{self.__optimized_alg_abbr} <======"
            logger_headline += f"\n|-> {self.__target_alg_abbr} target = {target_solution}"

            self.meta_ga.run_meta_ga(
                filename=self.__meta_ga_pkl_filename,
                target_algorithm=target_algorithm,
                prefix=str(comparison_idx),
                suffix=self.__comparison_dir_suffix,
                log_headline=logger_headline,
            )
            if self.meta_ga.meta_ga is not None:
                self.optimized_solutions.append(self.meta_ga.meta_ga.best_solutions[-1])

        print(f"\nAnalysis completed in: {timer(start, time.time())}")
        if generate_dataset:
            print("Generating dataset...")
            self.generate_dataset_from_solutions(num_runs=num_runs)
            if calculate_similarity_metrics:
                print("Calculating similarity metrics...")
                self.calculate_similarity_metrics()
        if export:
            print("Exporting .pkl file...")
            self.export_to_pkl(pkl_filename)

        print(f"\nAll done in: {timer(start, time.time())}")

    def export_to_pkl(self, filename):
        """
        Export instance of the metaheuristic similarity analyzer as .pkl.

        Args:
            filename (str): Filename of the output file. File extension .pkl included upon export.
        """
        self.__absolute_dirname = None
        filename = os.path.join(self.archive_path, filename)
        msa = cloudpickle.dumps(self)
        with open(filename + ".pkl", "wb") as file:
            file.write(msa)
            cloudpickle.dump(self, file)

    @staticmethod
    def import_from_pkl(filename) -> "MetaheuristicsSimilarityAnalyzer":
        """
        Import saved instance of the metaheuristic similarity analyzer.

        Args:
            filename (str): Filename of the file to import. File extension .pkl included upon import.

        Returns:
            msa (MetaheuristicSimilarityAnalyzer): Metaheuristic similarity analyzer instance.

        Raises:
            FileNotFoundError: File not found.
            BaseException: File could not be loaded.
            TypeError: Imported object is not a `MetaheuristicsSimilarityAnalyzer` instance.
        """

        try:
            with open(filename + ".pkl", "rb") as file:
                msa = cloudpickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename}.pkl not found.")
        except Exception:
            raise BaseException(f"File {filename}.pkl could not be loaded.")
        msa.__absolute_dirname = os.path.join(os.getcwd(), os.path.dirname(filename))
        if not isinstance(msa, MetaheuristicsSimilarityAnalyzer):
            raise TypeError("Provided .pkl file is not a `MetaheuristicsSimilarityAnalyzer` export.")
        return msa

    def __import_comparison_meta_ga(self, comparison_index: int):
        r"""Imports MetaGA object of the selected comparison.

        Args:
            comparison_index (int): Index of the comparison to create a plot for.

        Returns:
            (comparison_path, MetaGA) (tuple[str, MetaGA]): Path to the folder structure
                depending on the current context and the imported MetaGA object.

        Raises:
            ValueError: `comparison_index` out of range.
        """
        if comparison_index > len(self.optimized_solutions) - 1:
            raise ValueError(
                f"`comparison_index` {comparison_index} out of range [0, {len(self.optimized_solutions) - 1}]."
            )
        comparison_dir = "_".join([str(comparison_index), self.__comparison_dir_suffix])
        if self.__absolute_dirname is not None:
            comparison_path = os.path.join(self.__absolute_dirname, comparison_dir)
        else:
            comparison_path = os.path.join(self.archive_path, comparison_dir)
        imported_meta_ga = MetaGA.import_from_pkl(os.path.join(comparison_path, self.__meta_ga_pkl_filename))
        return comparison_path, imported_meta_ga

    def plot_solutions(
        self, comparison_index: int, filename: str = "meta_ga_solution_evolution", all_solutions: bool = False
    ):
        r"""Creates and shows a figure showing the solutions trough MetaGA generations.

        Args:
            comparison_index (int): Index of the comparison to create a plot for.
            filename (Optional[str]): Filename of the .png file saved.
                File is saved under the corresponding comparisons directory.
            all_solutions (Optional[str]): Plot evolution including all solutions.
                If false only the best solutions of each generation are plotted.

        Raises:
            ValueError: `comparison_index` out of range.
        """
        comparison_path, imported_meta_ga = self.__import_comparison_meta_ga(comparison_index)
        title = f"{comparison_index} comparison solutions"
        file_path = os.path.join(comparison_path, filename)
        imported_meta_ga.plot_solutions(title, file_path, all_solutions=all_solutions)

    def plot_fitness(self, comparison_index: int, filename: str = "meta_ga_fitness_plot"):
        r"""Creates and shows a figure showing the fitness trough MetaGA generations.

        Args:
            comparison_index (int): Index of the comparison to create a plot for.
            filename (Optional[str]): Filename of the .png file saved.
                File is saved under the corresponding comparisons directory.

        Raises:
            ValueError: `comparison_index` out of range.
        """
        comparison_path, imported_meta_ga = self.__import_comparison_meta_ga(comparison_index)
        file_path = os.path.join(comparison_path, filename)
        title = f"{comparison_index} comparison {self.__target_alg_abbr}-{self.__optimized_alg_abbr} Meta-GA Fitness"
        imported_meta_ga.plot_fitness(title, file_path)

    def indiv_diversity_metrics_comparison(self, comparison_index: int, run_index: int):
        r"""Creates and shows a figure showing the individual diversity metrics of both metaheuristics.

        Args:
            comparison_index (int): Index of the comparison to create a plot for.
            run_index (Optional[str]): Index of the optimization run to plot the metrics for.

        Raises:
            ValueError: `comparison_index` out of range.
            ValueError: `run_index` out of range.
        """
        if comparison_index > len(self.optimized_solutions) - 1:
            raise ValueError(
                f"`comparison_index` {comparison_index} out of range [0, {len(self.optimized_solutions) - 1}]."
            )
        comparison_dir = "_".join([str(comparison_index), self.__comparison_dir_suffix])
        if self.__absolute_dirname is not None:
            dataset_path = os.path.join(self.__absolute_dirname, self.__dataset_dirname, comparison_dir)
        else:
            dataset_path = os.path.join(self.archive_path, self.__dataset_dirname, comparison_dir)
        target_runs = get_sorted_list_of_runs(dataset_path, self.__target_alg_abbr)
        optimized_runs = get_sorted_list_of_runs(dataset_path, self.__optimized_alg_abbr)
        if run_index >= len(target_runs) or run_index >= len(optimized_runs):
            raise ValueError(f"`run_index` {str(run_index)} out of range [0, {len(target_runs) - 1}].")
        target_run = SingleRunData.import_from_json(target_runs[run_index])
        optimized_run = SingleRunData.import_from_json(optimized_runs[run_index])

        target_metrics = target_run.get_indiv_diversity_metrics_values()
        optimized_metrics = optimized_run.get_indiv_diversity_metrics_values()

        plots_per_line = 4
        num_metrics = len(self.meta_ga.indiv_diversity_metrics)
        lines = int((num_metrics - 1) / plots_per_line) + 1
        fig, axes = plt.subplots(lines, plots_per_line)
        title = f"{comparison_index} comparison run {run_index} individual diversity metrics"
        fig.suptitle(title, fontsize=23)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.subplots_adjust(wspace=0.4, hspace=0.4)

        for idx, metric in enumerate(self.meta_ga.indiv_diversity_metrics):
            df_target_metric = target_metrics.filter(regex=metric.abbreviation())
            df_target_metric.columns = df_target_metric.columns.str.replace(
                metric.abbreviation(), self.__target_alg_abbr
            )
            df_optimized_metric = optimized_metrics.filter(regex=metric.abbreviation())
            df_optimized_metric.columns = df_optimized_metric.columns.str.replace(
                metric.abbreviation(), self.__optimized_alg_abbr
            )
            df_combined_metrics = pd.concat([df_target_metric, df_optimized_metric], axis=1)
            ax = df_combined_metrics.plot(
                ax=axes[int(idx / plots_per_line)][idx % plots_per_line] if lines > 1 else axes[idx],
                kind="box",
                widths=0.5,
                figsize=(15, 5 * lines),
                fontsize=15,
            )
            ax.set_title(label=metric.abbreviation(), fontdict={"fontsize": 20})
        plt.show()

    def pop_diversity_metrics_comparison(
        self, comparison_index: int, run_index: int, title: str | None = None, separate: bool = False
    ):
        r"""Creates and shows a figure showing the population diversity metrics of both metaheuristics.

        Args:
            comparison_index (int): Index of the comparison to create a plot for.
            run_index (Optional[str]): Index of the optimization run to plot the metrics for.
            separate (Optional[bool]): Show diversity metrics on separate axes for better resolution.

        Raises:
            ValueError: `comparison_index` out of range.
            ValueError: `run_index` out of range.
        """
        if comparison_index > len(self.optimized_solutions) - 1:
            raise ValueError(
                f"`comparison_index` {comparison_index} out of range [0, {len(self.optimized_solutions) - 1}]."
            )
        comparison_dir = "_".join([str(comparison_index), self.__comparison_dir_suffix])
        if self.__absolute_dirname is not None:
            dataset_path = os.path.join(self.__absolute_dirname, self.__dataset_dirname, comparison_dir)
        else:
            dataset_path = os.path.join(self.archive_path, self.__dataset_dirname, comparison_dir)
        target_runs = get_sorted_list_of_runs(dataset_path, self.__target_alg_abbr)
        optimized_runs = get_sorted_list_of_runs(dataset_path, self.__optimized_alg_abbr)
        if run_index >= len(target_runs) or run_index >= len(optimized_runs):
            raise ValueError(f"`run_index` {str(run_index)} out of range [0, {len(target_runs) - 1}].")
        target_run = SingleRunData.import_from_json(target_runs[run_index])
        optimized_run = SingleRunData.import_from_json(optimized_runs[run_index])
        target_metrics = target_run.get_pop_diversity_metrics_values()
        optimized_metrics = optimized_run.get_pop_diversity_metrics_values()

        title = f"{comparison_index} comparison run {run_index} population diversity metrics"

        if separate:
            num_metrics = len(self.meta_ga.pop_diversity_metrics)
            fig, axes = plt.subplots(num_metrics, 1)
            fig.suptitle(title, fontsize=23)

            for idx, metric in enumerate(self.meta_ga.pop_diversity_metrics):
                df_target_metric = target_metrics.filter(regex=metric.abbreviation())
                df_target_metric.columns = df_target_metric.columns.str.replace(
                    metric.abbreviation(), self.__target_alg_abbr
                )
                df_optimized_metric = optimized_metrics.filter(regex=metric.abbreviation())
                df_optimized_metric.columns = df_optimized_metric.columns.str.replace(
                    metric.abbreviation(), self.__optimized_alg_abbr
                )
                df_combined_metrics = pd.concat([df_target_metric, df_optimized_metric], axis=1)
                ax = df_combined_metrics.plot(
                    ax=axes[idx],
                    kind="line",
                    figsize=(15, 3 * num_metrics),
                    fontsize=15,
                )
                ax.set_xlabel(xlabel="Iterations", fontdict={"fontsize": 15})
                ax.set_title(label=metric.abbreviation(), fontdict={"fontsize": 20})
            fig.tight_layout()
        else:
            target_metrics = target_metrics.add_suffix(f"_{self.__target_alg_abbr}")
            optimized_metrics = optimized_metrics.add_suffix(f"_{self.__optimized_alg_abbr}")
            df_combined_metrics = pd.concat([target_metrics, optimized_metrics], axis=1)
            line_styles = ["-", ":", "--", "-."]
            style = {}
            for idx, metric in enumerate(self.meta_ga.pop_diversity_metrics):
                if idx > 3:
                    continue
                style["_".join([metric.abbreviation(), self.__target_alg_abbr])] = f"{line_styles[idx]}g"
                style["_".join([metric.abbreviation(), self.__optimized_alg_abbr])] = f"{line_styles[idx]}b"

            fig, axes = plt.subplots(1, 1)
            fig.suptitle(title, fontsize=23)
            ax = df_combined_metrics.plot(ax=axes, style=style, figsize=(15, 5), logy=True, fontsize=15)
            ax.legend(fontsize=15)
            ax.set_xlabel(xlabel="Iterations", fontdict={"fontsize": 20})
            fig.tight_layout()
        plt.show()

    def svm_and_knn_classification_similarity_metrics(
        self,
        repetitions: int,
        get_train_accuracy: bool = False,
        bar_chart_filename: str | None = None,
        box_plot_filename: str | None = None,
    ):
        r"""Evaluate similarity of metaheuristics with SVM and KNN classifiers based on feature vectors.
        Based on assumption should models perform worse when distinguishing metaheuristics with higher similarity.
        To maximize metaheuristics similarity metric `1-accuracy` is used as similarity metric.

        Args:
            repetitions (int): Number of training repetitions to get the average 1-accuracy score from.
            get_train_accuracy (Optional[bool]): Return accuracy of models on train subset if True.
            bar_chart_filename (Optional[str]): Filename of the bar charts showing metric 1-accuracy values of the
                models per MetaheuristicSimilarityAnalyzer comparison.
            box_plot_filename (Optional[str]): Filename of the box plot showing metric 1-accuracy values of the models
                for all MetaheuristicSimilarityAnalyzer comparisons.

        Returns:
            1-accuracy scores (dict[str, numpy.ndarray[float]]): Dictionary containing 1-accuracy scores for test and
                also train (if `get_train_accuracy` is True) subsets of both models.
        """
        alg_1_label = self.__target_alg_abbr
        alg_2_label = self.__optimized_alg_abbr
        _k_svm_scores = []
        _knn_scores = []
        comparisons = os.listdir(self.dataset_path)

        for idx in range(len(comparisons)):
            comparison = f"{idx}_comparison"
            comparison_k_svm_scores = []
            comparison_knn_scores = []
            feature_vectors = []
            actual_labels = []

            first_runs = get_sorted_list_of_runs(os.path.join(self.dataset_path, comparison), self.__target_alg_abbr)
            second_runs = get_sorted_list_of_runs(
                os.path.join(self.dataset_path, comparison), self.__optimized_alg_abbr
            )
            for alg_label, runs in enumerate([first_runs, second_runs]):
                for run_path in runs:
                    srd = SingleRunData.import_from_json(run_path)
                    feature_vector = srd.get_feature_vector(standard_scale=True)
                    feature_vectors.append(feature_vector)
                    actual_labels.append(alg_label)

            for _ in range(repetitions):
                # train test split
                X_train, X_test, y_train, y_test = train_test_split(
                    feature_vectors, actual_labels, test_size=0.2, shuffle=True
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # K-SVM classifier
                # define the parameter grid
                param_grid = {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "gamma": [0.0001, 0.001, 0.01, 1, 10, 100, 1000],
                }

                k_svm = svm.SVC(kernel="rbf")
                kf = KFold(n_splits=5, shuffle=True, random_state=None)

                # perform grid search
                grid_search = GridSearchCV(k_svm, param_grid, cv=kf, n_jobs=-1)
                grid_search.fit(X_train, y_train)

                _C = grid_search.best_params_.get("C")
                _gamma = grid_search.best_params_.get("gamma")
                k_svm = svm.SVC(
                    kernel="rbf",
                    C=0 if _C is None else _C,
                    gamma=0 if _gamma is None else _gamma,
                )
                k_svm.fit(X_train, y_train)
                svm_training_score = k_svm.score(X_train, y_train)
                svm_test_score = k_svm.score(X_test, y_test)
                tmp = []
                tmp.append(1.0 - svm_training_score)
                tmp.append(1.0 - svm_test_score)
                comparison_k_svm_scores.append(tmp)

                # kNN classifier
                # define the parameter grid
                param_grid = {"n_neighbors": np.arange(1, min(50, round(len(y_train) / 3))).tolist()}

                knn = KNeighborsClassifier()
                kf = KFold(n_splits=5, shuffle=True, random_state=None)

                # perform grid search
                grid_search = GridSearchCV(knn, param_grid, cv=kf, n_jobs=-1)
                grid_search.fit(X_train, y_train)

                _n_neighbors = grid_search.best_params_.get("n_neighbors")
                if _n_neighbors is None:
                    _n_neighbors = 0
                knn = KNeighborsClassifier(n_neighbors=_n_neighbors)
                knn.fit(X_train, y_train)
                knn_training_score = knn.score(X_train, y_train)
                knn_test_score = knn.score(X_test, y_test)
                tmp = []
                tmp.append(1.0 - knn_training_score)
                tmp.append(1.0 - knn_test_score)
                comparison_knn_scores.append(tmp)

            _k_svm_scores.append(np.mean(comparison_k_svm_scores, axis=0))
            _knn_scores.append(np.mean(comparison_knn_scores, axis=0))

        k_svm_scores = np.array(_k_svm_scores)
        knn_scores = np.array(_knn_scores)

        # bar charts
        if bar_chart_filename is not None:
            bar_width = 0.35
            (
                fig,
                ax,
            ) = plt.subplots(2, 1, figsize=(15, 10))
            fig.subplots_adjust(hspace=0.5)
            index = np.arange(1, len(k_svm_scores[:, 0]) + 1)
            low = np.min(k_svm_scores)
            high = np.max(k_svm_scores)
            ax[0].bar(index, k_svm_scores[:, 0], bar_width, label="train")
            ax[0].bar(index + bar_width, k_svm_scores[:, 1], bar_width, label="test")
            ax[0].set_title(f"SVM {alg_1_label} - {alg_2_label}", fontsize=22, pad=10)
            ax[0].set_xlabel("configuration", fontsize=19, labelpad=10)
            ax[0].set_ylabel("1-accuracy", fontsize=19, labelpad=10)
            ax[0].tick_params(axis="x", labelsize=19, rotation=45)
            ax[0].tick_params(axis="y", labelsize=19)
            ax[0].legend(fontsize=15)
            ax[0].xaxis.set_ticks(index + bar_width / 2, index)
            ax[0].set_ylim(low - 0.5 * (high - low), high + 0.5 * (high - low))
            ax[0].set_xlim(
                ax[0].patches[0].get_x() / 2,
                ax[0].patches[-1].get_x() + ax[0].patches[-1].get_width() * 2,
            )
            ax[0].grid(axis="y", color="gray", linestyle="--", linewidth=0.7)
            ax[0].set_axisbelow(True)

            low = np.min(knn_scores)
            high = np.max(knn_scores)
            ax[1].bar(index, knn_scores[:, 0], bar_width, label="train")
            ax[1].bar(index + bar_width, knn_scores[:, 1], bar_width, label="test")
            ax[1].set_title(f"KNN {alg_1_label} - {alg_2_label}", fontsize=22, pad=10)
            ax[1].set_xlabel("configuration", fontsize=19, labelpad=10)
            ax[1].set_ylabel("1-accuracy", fontsize=19, labelpad=10)
            ax[1].tick_params(axis="x", labelsize=19, rotation=45)
            ax[1].tick_params(axis="y", labelsize=19)
            ax[1].legend(fontsize=15)
            ax[1].xaxis.set_ticks(index + bar_width / 2, index)
            ax[1].set_ylim(low - 0.5 * (high - low), high + 0.5 * (high - low))
            ax[1].set_xlim(
                ax[1].patches[0].get_x() / 2,
                ax[1].patches[-1].get_x() + ax[1].patches[-1].get_width() * 2,
            )
            ax[1].grid(axis="y", color="gray", linestyle="--", linewidth=0.7)
            ax[1].set_axisbelow(True)

            fig.savefig(bar_chart_filename, bbox_inches="tight")

        # box plots
        if box_plot_filename is not None:
            (
                fig,
                ax,
            ) = plt.subplots(1, 2, figsize=(15, 5))
            fig.subplots_adjust(wspace=0.3)
            labels = ["train", "test"]

            ax[0].boxplot(k_svm_scores)
            ax[0].set_xticks(ticks=np.arange(1, len(labels) + 1), labels=labels)
            ax[0].tick_params(axis="both", labelsize=19)
            ax[0].set_title(f"SVM  {alg_1_label} - {alg_2_label}", fontsize=22, pad=15)
            ax[0].tick_params(axis="both", labelsize=19)
            ax[0].set_ylabel("1-accuracy", fontsize=19, labelpad=10)

            ax[1].boxplot(knn_scores)
            ax[1].set_xticks(ticks=np.arange(1, len(labels) + 1), labels=labels)
            ax[1].tick_params(axis="both", labelsize=19)
            ax[1].set_title(f"KNN  {alg_1_label} - {alg_2_label}", fontsize=22, pad=15)
            ax[1].tick_params(axis="both", labelsize=19)
            ax[1].set_ylabel("1-accuracy", fontsize=19, labelpad=10)

            fig.savefig(box_plot_filename, bbox_inches="tight")

        accuracy = {
            SimilarityMetrics.SVM.value: np.round(k_svm_scores.transpose()[1], 2).tolist(),
            SimilarityMetrics.KNN.value: np.round(knn_scores.transpose()[1], 2).tolist(),
        }
        if get_train_accuracy:
            train_accuracy = {
                "svm_train": np.round(k_svm_scores.transpose()[0], 2).tolist(),
                "knn_train": np.round(knn_scores.transpose()[0], 2).tolist(),
            }
            accuracy.update(train_accuracy)
        return accuracy

    def export_results_to_latex(self, filename: str | None = None, generate_pdf: bool = False):
        r"""Generate latex file containing MSA results in form of tables.
        Optionally also generate pdf file.

        Args:
            filename (Optional[str]): Filename without extensions used for the .tex and .pdf files.
                Composed if not provided.
            generate_pdf (Optional[bool]): Generates a .pdf file. Only .tex file is generated if false.
        """

        if len(self.similarity_metrics) == 0:
            warnings.warn(
                """Similarity metrics were not calculated and thus can not be displayed!
                    To calculate similarity metrics call method `calculate_similarity_metrics`!"""
            )

        geometry_options = {
            "inner": "3.5cm",
            "outer": "2.5cm",
            "top": "3.0cm",
            "bottom": "3.0cm",
        }
        doc = Document(
            geometry_options=geometry_options,
            documentclass="book",
            document_options=["openany", "a4paper", "12pt", "fleqn"],
        )
        doc.packages.append(Package("makecell"))
        doc.packages.append(Package("array"))
        doc.packages.append(Package("multirow"))
        doc.packages.append(Package("rotating"))
        doc.packages.append(Package("graphicx"))
        doc.packages.append(Package("amsmath"))

        doc.append(Section(f"Comparison of {self.__target_alg_abbr} and {self.__optimized_alg_abbr}"))
        doc.append(Subsection("Comparison of hyperparameters settings and similarity metrics"))

        # Table comparing hyperparameters settings and similarity metrics
        if len(self.similarity_metrics) != 0:
            hyperparameters_table = self.get_hyperparameters_and_similarity_metrics_latex_table()
            doc.append(hyperparameters_table)
        else:
            doc.append("Not able to display the table because similarity metrics were not calculated.")

        doc.append(Subsection("Comparison of fitness statistics"))

        # Table comparing fitness statistics
        fitness_table = self.get_fitness_comparison_latex_table()
        doc.append(fitness_table)

        if self.__absolute_dirname is not None:
            archive_path = self.__absolute_dirname
        else:
            archive_path = self.archive_path
        if filename is None:
            filename = f"{self.__target_alg_abbr}-{self.__optimized_alg_abbr}_MSA_results"
        if generate_pdf:
            doc.generate_pdf(
                os.path.join(
                    archive_path,
                    filename,
                ),
                clean_tex=False,
            )
        else:
            doc.generate_tex(
                os.path.join(
                    archive_path,
                    filename,
                )
            )

    def get_hyperparameters_and_similarity_metrics_latex_table(self):
        r"""Create latex table displaying hyperparameters settings and similarity metrics
            of target and optimized metaheuristic.

        Returns:
            hyperparameters_table (LongTable): Table displaying hyperparameters settings
                and similarity metrics of compared metaheuristics.
        """

        # Create table header
        table_specs = (
            "p{0.8cm} |"
            + " c" * len(self.target_gene_space[next(iter(self.target_gene_space))])
            + " |"
            + " c" * len(self.meta_ga.gene_space[next(iter(self.meta_ga.gene_space))])
            + " |"
            + " c" * len(self.similarity_metrics)
        )
        table = LongTable(table_specs)
        mc_target = MultiColumn(
            len(self.target_gene_space[next(iter(self.target_gene_space))]),
            align="c|",
            data=self.__target_alg_abbr,
        )
        mc_optimized = MultiColumn(
            len(self.meta_ga.gene_space[next(iter(self.meta_ga.gene_space))]),
            align="c|",
            data=self.__optimized_alg_abbr,
        )
        mc_metrics = MultiColumn(
            len(self.similarity_metrics),
            data="Similarity Metrics",
        )

        table.add_hline()
        table.add_row(["", mc_target, mc_optimized, mc_metrics])

        cells = [NoEscape(r"\rotatebox{0}{c.n.} ")]
        for alg_name in self.target_gene_space:
            for setting in self.target_gene_space[alg_name]:
                cells.append(NoEscape(r" \rotatebox{90}{\makecell{" + setting.replace("_", "-") + "}} "))
        for alg_name in self.meta_ga.gene_space:
            for setting in self.meta_ga.gene_space[alg_name]:
                cells.append(NoEscape(r" \rotatebox{90}{\makecell{" + setting.replace("_", "-") + "}} "))

        cells += [
            NoEscape(r" \rotatebox{90}{\makecell{$Sim_{\mathit{SMAPE}}$}} "),
            NoEscape(r" \rotatebox{90}{\makecell{$Sim_{\mathit{cos}}$}} "),
            NoEscape(r" \rotatebox{90}{\makecell{$\rho$}} "),
            NoEscape(r" \rotatebox{90}{\makecell{$Sim_{\mathit{SVM}}$}} "),
            NoEscape(r" \rotatebox{90}{\makecell{$Sim_{\mathit{KNN}}$}} "),
        ]

        table.add_hline()
        table.add_row(cells)
        table.add_hline()

        displayed_similarity_metrics = [
            self.similarity_metrics[SimilarityMetrics.SMAPE.value],
            self.similarity_metrics[SimilarityMetrics.COS.value],
            self.similarity_metrics[SimilarityMetrics.SPEARMAN.value],
            self.similarity_metrics[SimilarityMetrics.SVM.value],
            self.similarity_metrics[SimilarityMetrics.KNN.value],
        ]

        # Add rows
        for idx, (
            target,
            optimized,
            smape,
            cosine,
            rho,
            svm_test,
            knn_test,
        ) in enumerate(
            zip(
                self.target_solutions,
                self.optimized_solutions,
                *displayed_similarity_metrics,
            )
        ):
            cells = [f"{idx + 1}"]
            for t in target:
                cells.append(f"{round(t, 2)}")
            for o in optimized:
                cells.append(f"{round(o, 2)}")

            for value, list in zip(
                [smape, cosine, rho, svm_test, knn_test],
                displayed_similarity_metrics,
            ):
                if round(value, 2) == round(np.max(list), 2):
                    cells.append(bold(f"{round(value, 2)}"))
                else:
                    cells.append(f"{round(value, 2)}")
            table.add_row(cells)

        table.add_hline()

        # Calculate statistics at the end of the table
        for stat in [np.min, np.mean, np.max, np.std]:
            stats_row = np.concatenate(
                (
                    np.array([f"{stat.__name__}."]),
                    np.round(stat(np.array(self.target_solutions), axis=0), 2),
                    np.round(
                        stat(np.array(self.optimized_solutions), axis=0),
                        2,
                    ),
                )
            )
            for list in displayed_similarity_metrics:
                stats_row = np.append(stats_row, round(stat(list), 2))
            table.add_row(stats_row)

        table.add_hline()
        return table

    def get_fitness_comparison_latex_table(self):
        r"""Create latex table displaying statistics of fitness of target and optimized algorithm.

        Returns:
            fitness_table (LongTable): Table displaying statistics of fitness.
        """
        # Create table header
        fitness_table = LongTable("p{1cm} | c  c  c | c  c  c")
        mc_target = MultiColumn(
            3,
            align="c|",
            data=self.__target_alg_abbr,
        )
        mc_optimized = MultiColumn(
            3,
            data=self.__optimized_alg_abbr,
        )

        fitness_table.add_hline()
        fitness_table.add_row(["", mc_target, mc_optimized])

        fitness_table.add_hline()
        fitness_table.add_row(("c.n.", "min.", "mean.", "std.", "min.", "mean.", "std."))
        fitness_table.add_hline()

        if self.__absolute_dirname is not None:
            current_dataset_path = os.path.join(self.__absolute_dirname, self.__dataset_dirname)
        else:
            current_dataset_path = os.path.join(self.dataset_path)
        comparisons = os.listdir(current_dataset_path)

        # Collect fitness data from metaheuristic optimization runs
        fitness_statistics = []

        for alg_abbr in (self.__target_alg_abbr, self.__optimized_alg_abbr):
            mean_fitness = []
            min_fitness = []
            std_fitness = []
            for idx in range(len(comparisons)):
                comparison = "_".join([str(idx), self.__comparison_dir_suffix])
                comparison_dataset_path = os.path.join(current_dataset_path, comparison)
                runs = get_sorted_list_of_runs(comparison_dataset_path, alg_abbr)
                fitness = []
                for run_path in runs:
                    srd = SingleRunData.import_from_json(run_path)
                    fitness.append(srd.best_fitness)

                min_fitness.append(round(np.amin(fitness), 2))
                mean_fitness.append(round(np.mean(fitness), 2))
                std_fitness.append(round(np.std(fitness), 2))

            fitness_statistics.extend([min_fitness, mean_fitness, std_fitness])

        # Add rows
        for idx, (min_1, mean_1, std_1, min_2, mean_2, std_2) in enumerate(zip(*fitness_statistics)):
            row = [f"{idx + 1}"]
            for value, list in zip(
                [min_1, mean_1, std_1, min_2, mean_2, std_2],
                fitness_statistics,
            ):
                if value == np.min(list):
                    row.append(bold(f"{value}"))
                else:
                    row.append(f"{value}")
            fitness_table.add_row(row)
        fitness_table.add_hline()

        # Calculate statistics at the end of the table
        for stat in [np.min, np.mean, np.max, np.std]:
            row = [f"{stat.__name__}."]
            for data in fitness_statistics:
                row.append(round(stat(data), 2))
            fitness_table.add_row(row)

        fitness_table.add_hline()
        return fitness_table

    def msa_info(
        self,
        filename: str = "msa_info",
        table_background_color: str = "white",
        table_border_color: str = "black",
        graph_color: str = "grey",
        sub_graph_color: str = "lightgrey",
    ):
        r"""Produces a scheme of the MetaheuristicsSimilarityAnalyzer configuration.

        Args:
            filename (Optional[str]): Name of the scheme image file.
            table_background_color (Optional[str]): Table background color.
            table_border_color (Optional[str]): Table border color.
            graph_color (Optional[str]): Graph background color.
            sub_graph_color (Optional[str]): Sub graph background color.
        """

        gv = graphviz.Digraph("msa_info", filename=filename)
        gv.attr(rankdir="TD", compound="true")
        gv.attr("node", shape="box")
        gv.attr("graph", fontname="bold")
        gv.attr("graph", splines="false")

        with gv.subgraph(name="cluster_0") as c:
            c.attr(
                style="filled",
                color=graph_color,
                name="msa",
                label="Metaheuristics Similarity Analyzer",
            )
            c.node_attr.update(
                style="filled",
                color=table_border_color,
                fillcolor=table_background_color,
                shape="plaintext",
                margin="0",
            )
            msa_parameters_label = f"""<
                <table border="0" cellborder="1" cellspacing="0">
                    <tr>
                        <td colspan="2"><b>Parameters</b></td>
                    </tr>
                    <tr>
                        <td>target solutions</td>
                        <td>{len(self.target_solutions)}</td>
                    </tr>
                    <tr>
                        <td>runs per solutions</td>
                        <td>{self.meta_ga.num_runs}</td>
                    </tr>
                </table>>"""
            c.node(name="msa_parameters", label=msa_parameters_label)

            with c.subgraph(name="cluster_00") as cc:
                cc.attr(
                    style="filled",
                    color=sub_graph_color,
                    name="target_algorithm",
                    label="Target Algorithm",
                )
                cc.node_attr.update(
                    style="filled",
                    color=table_border_color,
                    fillcolor=table_background_color,
                    shape="plaintext",
                    margin="0",
                )

                target_parameters_len = 0
                for alg_name in self.target_gene_space:
                    algorithm = get_algorithm_by_name(alg_name)

                    node_label = f"""<<table border="0" cellborder="1" cellspacing="0">
                        <tr>
                            <td colspan="2"><b>{algorithm.Name[1]}</b></td>
                        </tr>
                        <tr>
                            <td>pop size</td>
                            <td>{self.meta_ga.pop_size}</td>
                        </tr>"""
                    for setting in self.target_gene_space[alg_name]:
                        gene = ", ".join(str(value) for value in self.target_gene_space[alg_name][setting].values())
                        node_label += f"<tr><td>{setting}</td><td>[{gene}]</td></tr>"
                        target_parameters_len += 1
                    node_label += "</table>>"
                    cc.node(name="target_gene_space", label=node_label)

                target_parameters = f"""<
                    <table border="0" cellborder="1" cellspacing="0">
                        <tr>
                            <td colspan="3"><b>Parameters</b></td>
                        </tr>
                        <tr>
                            <td>p<i><sub>1</sub></i></td>
                            <td>...</td>
                            <td>p<i><sub>{target_parameters_len}</sub></i></td>
                        </tr>
                    </table>>"""
                cc.node(name="target_parameters", label=target_parameters)

                cc.edge(
                    "target_gene_space",
                    "target_parameters",
                    label="target solution",
                )

        with gv.subgraph(name="cluster_1") as c:
            c.attr(
                style="filled",
                color=graph_color,
                name="meta_ga",
                label="Meta-GA",
            )
            c.node_attr.update(
                style="filled",
                color=table_border_color,
                fillcolor=table_background_color,
                shape="plaintext",
                margin="0",
            )
            meta_ga_parameters_label = f"""<
                <table border="0" cellborder="1" cellspacing="0">
                    <tr>
                        <td colspan="2"><b>Parameters</b></td>
                    </tr>
                    <tr>
                        <td>generations</td>
                        <td>{self.meta_ga.ga_generations}</td>
                    </tr>
                    <tr>
                        <td>pop size</td>
                        <td>{self.meta_ga.ga_solutions_per_pop}</td>
                    </tr>
                    <tr>
                        <td>parent selection</td>
                        <td>{self.meta_ga.ga_parent_selection_type}</td>
                    </tr>
                    """

            if self.meta_ga.ga_parent_selection_type == "tournament":
                meta_ga_parameters_label += f"""
                    <tr>
                        <td>K tournament</td>
                        <td>{self.meta_ga.ga_k_tournament}</td>
                    </tr>"""
            meta_ga_parameters_label += f"""
                    <tr>
                        <td>parents</td>
                        <td>{self.meta_ga.ga_percent_parents_mating} %</td>
                    </tr>
                    <tr>
                        <td>crossover type</td>
                        <td>{self.meta_ga.ga_crossover_type}</td>
                    </tr>
                    <tr>
                        <td>mutation type</td>
                        <td>{self.meta_ga.ga_mutation_type}</td>
                    </tr>
                    <tr>
                        <td>crossover prob.</td>
                        <td>{self.meta_ga.ga_crossover_probability}</td>
                    </tr>
                    <tr>
                        <td>mutate num genes</td>
                        <td>{self.meta_ga.ga_mutation_num_genes}</td>
                    </tr>
                    <tr>
                        <td>keep elitism</td>
                        <td>{self.meta_ga.ga_keep_elitism}</td>
                    </tr>
                    <tr>
                        <td>rng seed</td>
                        <td>run index</td>
                    </tr>
                </table>>"""

            c.node_attr.update(
                style="filled",
                color=table_border_color,
                fillcolor=table_background_color,
                shape="box",
            )

            c.node(
                name="sim_smape",
                label="Sim_SMAPE",
                color=table_border_color,
                margin="0.1,0,0.1,0",
            )

            c.node(name="meta_ga_parameters", label=meta_ga_parameters_label)

            with c.subgraph(name="cluster_10") as cc:
                cc.attr(
                    style="filled",
                    color=sub_graph_color,
                    name="optimized_algorithm",
                    label="Optimized Algorithm",
                )
                cc.node_attr.update(
                    style="filled",
                    color=table_border_color,
                    fillcolor=table_background_color,
                    shape="plaintext",
                    margin="0",
                )
                combined_gene_space_len = 0
                for alg_idx, alg_name in enumerate(self.meta_ga.gene_space):
                    algorithm = get_algorithm_by_name(alg_name)
                    node_label = f"""<<table border="0" cellborder="1" cellspacing="0">
                        <tr>
                            <td colspan="2"><b>{algorithm.Name[1]}</b></td>
                        </tr>
                        <tr>
                            <td>pop size</td>
                            <td>{self.meta_ga.pop_size}</td>
                        </tr>"""
                    for setting in self.meta_ga.gene_space[alg_name]:
                        gene = ", ".join(str(value) for value in self.meta_ga.gene_space[alg_name][setting].values())
                        combined_gene_space_len += 1
                        node_label += (
                            f"<tr><td>{setting}</td><td>[{gene}]<sub> g<i>{combined_gene_space_len}</i></sub></td></tr>"
                        )
                    node_label += "</table>>"
                    cc.node(name=f"gene_space_{alg_idx}", label=node_label)

                combined_gene_string = f"""<
                <table border="0" cellborder="1" cellspacing="0">
                    <tr>
                        <td colspan="3"><b>Solution</b></td>
                        <td><b>Fitness</b></td>
                    </tr>
                    <tr>
                        <td>g<i><sub>1</sub></i></td>
                        <td>...</td>
                        <td>g<i><sub>{combined_gene_space_len}</sub></i></td>
                        <td port="gene_fitness">?</td>
                    </tr>
                </table>>"""
                cc.node(name="combined_gene_space", label=combined_gene_string)

                for alg_idx in range(len(self.meta_ga.gene_space)):
                    cc.edge(f"gene_space_{alg_idx}", "combined_gene_space")

            c.edge("sim_smape", "combined_gene_space:gene_fitness")

        with gv.subgraph(name="cluster_2") as c:
            c.attr(
                style="filled",
                color=graph_color,
                name="optimization",
                label="Optimization",
            )
            c.node_attr.update(
                style="filled",
                color=table_border_color,
                fillcolor=table_background_color,
                shape="plaintext",
                margin="0",
            )
            c.node(
                name="optimization_parameters",
                label=f"""<
                <table border="0" cellborder="1" cellspacing="0">
                    <tr>
                        <td colspan="2"><b>Parameters</b></td>
                    </tr>
                    <tr>
                        <td>max evals</td>
                        <td>{self.meta_ga.max_evals}</td>
                    </tr>
                    <tr>
                        <td>num runs</td>
                        <td>{self.meta_ga.num_runs}</td>
                    </tr>
                    <tr>
                        <td>problem</td>
                        <td>{self.meta_ga.problem.name()}</td>
                    </tr>
                    <tr>
                        <td>dimension</td>
                        <td>{self.meta_ga.problem.dimension}</td>
                    </tr>
                </table>>""",
            )

            with c.subgraph(name="cluster_20") as cc:
                cc.attr(
                    style="filled",
                    color=sub_graph_color,
                    name="metrics",
                    label="Diversity Metrics",
                )
                cc.node_attr.update(
                    style="filled",
                    color=table_border_color,
                    fillcolor=table_background_color,
                    shape="plaintext",
                    margin="0",
                )
                pop_metrics_label = (
                    '<<table border="0" cellborder="1" cellspacing="0"><tr><td><b>Pop Metrics</b></td></tr>'
                )
                if self.meta_ga.pop_diversity_metrics is not None:
                    for pop_metric in self.meta_ga.pop_diversity_metrics:
                        pop_metrics_label += f"""<tr><td>{pop_metric.abbreviation()}</td></tr>"""
                pop_metrics_label += "</table>>"
                cc.node(name="pop_metrics", label=pop_metrics_label)

                indiv_metrics_label = (
                    '<<table border="0" cellborder="1" cellspacing="0"><tr><td><b>Indiv Metrics</b></td></tr>'
                )
                if self.meta_ga.indiv_diversity_metrics is not None:
                    for indiv_metric in self.meta_ga.indiv_diversity_metrics:
                        indiv_metrics_label += f"""<tr><td>{indiv_metric.abbreviation()}</td></tr>"""
                indiv_metrics_label += "</table>>"
                cc.node(name="indiv_metrics", label=indiv_metrics_label)

        with gv.subgraph(name="cluster_3") as c:
            c.attr(
                style="dashed",
                color=table_border_color,
                name="population",
                label="Single Optimization Run",
            )
            c.node_attr.update(
                style="filled",
                color=table_border_color,
                fillcolor=table_background_color,
                shape="plaintext",
                margin="0",
            )

            pop_size = self.meta_ga.pop_size
            max_iters = self.meta_ga.max_iters

            c.node(
                name="pop_scheme",
                label=f"""<
                <table border="0" cellborder="0" cellspacing="0">
                    <tr>
                        <td>
                            <table border="0" cellborder="1" cellspacing="10">
                                <tr>
                                    <td><i><b>X</b><sub>i=1, t=1</sub></i></td>
                                    <td>...</td>
                                    <td><i><b>X</b><sub>i=1, t={max_iters}</sub></i></td>
                                </tr>
                                <tr>
                                    <td>...</td>
                                    <td>...</td>
                                    <td>...</td>
                                </tr>
                                <tr>
                                    <td><i><b>X</b><sub>i={pop_size}, t=1</sub></i></td>
                                    <td>...</td>
                                    <td><i><b>X</b><sub>i={pop_size}, t={max_iters}</sub></i></td>
                                </tr>
                            </table>
                        </td>
                        <td>
                            <table border="0" cellborder="0" cellspacing="10">
                                <tr>
                                    <td><i><b>IM</b><sub>1</sub></i></td>
                                </tr>
                                <tr>
                                    <td>...</td>
                                </tr>
                                <tr>
                                    <td><i><b>IM</b><sub>{pop_size}</sub></i></td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <table border="0" cellborder="0" cellspacing="0">
                                <tr>
                                    <td><i><b>PM</b><sub>1</sub></i></td>
                                    <td>...</td>
                                    <td><i><b>PM</b><sub>{max_iters}</sub></i></td>
                                </tr>
                            </table>
                        </td>
                        <td></td>
                    </tr>
                </table>>""",
            )

        gv.edge(
            minlen="3",
            tail_name="combined_gene_space",
            head_name="pop_metrics",
            xlabel="for each \nsolution",
            lhead="cluster_2",
        )
        gv.edge(
            dir="both",
            minlen="2",
            tail_name="optimization_parameters",
            head_name="pop_scheme",
            ltail="cluster_2",
            lhead="cluster_3",
        )
        gv.edge(
            tail_name="target_parameters",
            head_name="optimization_parameters",
            label="for each target \nsolution",
            lhead="cluster_2",
        )
        gv.edge(
            tail_name="pop_metrics",
            head_name="sim_smape",
            label="average feature vectors \nof target and optimized \nalgorithms \ndiversity metrics",
            ltail="cluster_2",
        )

        gv.attr(fontsize="25")

        gv.render(format="png", cleanup=True)
