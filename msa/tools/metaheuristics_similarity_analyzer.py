from datetime import datetime
from pathlib import Path
import warnings
import os
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
from msa.util.helper import random_float_with_step, get_algorithm_by_name
from msa.tools.meta_ga import MetaGA, MetaGAFitnessFunction
from msa.tools.optimization_tools import optimization_runner
from msa.tools.optimization_data import SingleRunData
import numpy as np
import numpy.typing as npt
from scipy import spatial, stats
import graphviz
import cloudpickle
from niapy.algorithms import Algorithm
from enum import Enum

__all__ = ["MetaheuristicsSimilarityAnalyzer", "SimilarityMetrics"]


class SimilarityMetrics(Enum):
    SMAPE = "smape"
    COS = "cosine"
    SPEARMAN = "spearman_r"
    KNN = "knn_test"
    SVM = "svm_test"


class MetaheuristicsSimilarityAnalyzer:
    r"""Class for search and analysis of similarity of metaheuristics with
    different parameter settings. Uses target metaheuristic with stochastically
    selected parameters and aims to find parameters of the optimized
    metaheuristic with which they perform in a similar maner.
    """

    def __init__(
        self,
        meta_ga: MetaGA,
        target_gene_space: dict[str | Algorithm, dict[str, dict[str, float]]],
        base_archive_path: str = "archive",
    ) -> None:
        r"""Initialize the metaheuristic similarity analyzer.

        Args:
            meta_ga (Optional[MetaGA]): Preconfigured instance of the meta
                genetic algorithm with fitness function set to
                `TARGET_PERFORMANCE_SIMILARITY`.
            target_gene_space (dict[str | Algorithm, dict[str, dict[str, float]]]):
                Gene space of the reference metaheuristic.
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
        self.target_alg_abbr = get_algorithm_by_name(list(target_gene_space)[0]).Name[1]
        self.optimized_alg_abbr = get_algorithm_by_name(list(self.meta_ga.gene_spaces)[0]).Name[1]
        self._base_archive_path = base_archive_path

    def __generate_targets(self, num_comparisons: int, generate_optimized_targets: bool = False):
        r"""Generate target solutions.

        Args:
            generate_optimized_targets (Optional[bool]): Generate target
                solutions by parameter tuning if True, otherwise generate
                random targets.
            num_comparisons (Optional[int]): Number of metaheuristic parameter
                combinations to analyze during the similarity analysis.

        Raises:
            ValueError: Algorithm does not have the attribute provided in the `gene_spaces`.
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

        self.__create_folder_structure()

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
                    gene_spaces=self.target_gene_space,
                    pop_size=self.meta_ga.pop_size,
                    max_evals=self.meta_ga.max_evals,
                    num_runs=self.meta_ga.num_runs,
                    problem=self.meta_ga.problem,
                    base_archive_path=os.path.join(self.archive_path, "target_tuning"),
                )
                target_solution = meta_ga.run_meta_ga(prefix=str(idx), return_best_solution=True)
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

        subsets = os.listdir(self.dataset_path)

        mean_smape = []
        cosine_similarity = []
        spearman_r = []

        problem = self.meta_ga.problem.name()

        for idx in range(len(subsets)):
            subset = f"{idx}_subset"
            feature_vectors_1 = []
            feature_vectors_2 = []

            first_runs = os.listdir(os.path.join(self.dataset_path, subset, self.target_alg_abbr, problem))
            second_runs = os.listdir(
                os.path.join(
                    self.dataset_path,
                    subset,
                    self.optimized_alg_abbr,
                    problem,
                )
            )

            first_runs.sort()
            second_runs.sort()

            smape_values = []
            for fr, sr in zip(first_runs, second_runs):
                first_run_path = os.path.join(
                    self.dataset_path,
                    subset,
                    self.target_alg_abbr,
                    problem,
                    fr,
                )
                second_run_path = os.path.join(
                    self.dataset_path,
                    subset,
                    self.optimized_alg_abbr,
                    problem,
                    sr,
                )
                f_srd = SingleRunData.import_from_json(first_run_path)
                s_srd = SingleRunData.import_from_json(second_run_path)

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
                metaheuristic for each solution. if None value assigned
                to meta genetic algorithm is used.
        """
        if num_runs is None:
            num_runs = self.meta_ga.num_runs

        for idx, (solution_0, solution_1) in enumerate(zip(self.target_solutions, self.optimized_solutions)):
            _subset_path = os.path.join(self.dataset_path, f"{idx}_subset")
            if os.path.exists(_subset_path) is False:
                Path(_subset_path).mkdir(parents=True, exist_ok=True)

            solution = np.append(solution_0, solution_1)
            gene_spaces = self.target_gene_space | self.meta_ga.gene_spaces
            algorithms = MetaGA.solution_to_algorithm_attributes(
                solution=solution,
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
                    max_evals=self.meta_ga.max_evals,
                    run_index_seed=True,
                    keep_pop_data=False,
                    parallel_processing=True,
                )

    def __create_folder_structure(self):
        r"""Create folder structure for metaheuristic similarity analysis."""
        datetime_now = str(datetime.now().strftime("%m-%d_%H.%M.%S"))
        self.archive_path = os.path.join(
            self._base_archive_path,
            "_".join(
                [
                    datetime_now,
                    self.target_alg_abbr,
                    self.meta_ga.problem.name(),
                ]
            ),
        )
        self.dataset_path = os.path.join(self.archive_path, "dataset")
        if os.path.exists(self.archive_path) is False:
            Path(self.archive_path).mkdir(parents=True, exist_ok=True)

    def run_similarity_analysis(
        self,
        num_comparisons: int | None = None,
        target_solutions: list[npt.NDArray] | None = None,
        generate_optimized_targets: bool = False,
        get_info: bool = False,
        generate_dataset: bool = False,
        calculate_similarity_metrics: bool = False,
        export: bool = False,
        pkl_filename: str = "msa_obj",
    ):
        r"""Run metaheuristic similarity analysis.

        Args:
            num_comparisons (Optional[int]): Number of metaheuristic parameter
                combinations to analyze during the similarity analysis.
                Required if `target_solutions` is None.
            target_solutions (Optional[list[numpy.ndarray]]): Target solutions
                for the target algorithm. Generated if None.
            generate_optimized_targets (Optional[bool]): Generate target
                solutions by parameter tuning. Target solutions wil be
                generated by uniform rng if false. Has no effect if
                `target_solutions` is not None.
            get_info (Optional[bool]): Generate info scheme of the
                metaheuristic similarity analyzer (false by default).
            generate_dataset (Optional[bool]): Generate dataset from
                target and optimized solutions after analysis
                (false by default).
            calculate_similarity_metrics (Optional[bool]): Calculates
                similarity metrics from target and optimized solutions
                after analysis (false by default). Has no effect if
                `generate_dataset` is false.
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
            self.__generate_targets(num_comparisons, generate_optimized_targets)
        elif target_solutions is not None:
            self.target_solutions = target_solutions
            self.__create_folder_structure()

        self.meta_ga.base_archive_path = self.archive_path

        if get_info:
            self.msa_info(
                filename=os.path.join(self.archive_path, "msa_info"),
            )

        for idx, target_solution in enumerate(self.target_solutions):
            target_algorithm = MetaGA.solution_to_algorithm_attributes(
                solution=target_solution,
                gene_spaces=self.target_gene_space,
                pop_size=self.meta_ga.pop_size,
            )
            self.meta_ga.run_meta_ga(target_algorithm=target_algorithm[0], prefix=str(idx))
            if self.meta_ga.meta_ga is not None:
                self.optimized_solutions.append(self.meta_ga.meta_ga.best_solutions[-1])

        if generate_dataset:
            self.generate_dataset_from_solutions()
            if calculate_similarity_metrics:
                print("Calculating similarity metrics...")
                self.calculate_similarity_metrics()
        if export:
            print("Exporting .pkl file...")
            self.export_to_pkl(pkl_filename)

        print("All done!")

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

        Raises:
            FileNotFoundError: File not found.
            BaseException: File could not be loaded.
        """

        try:
            with open(filename + ".pkl", "rb") as file:
                msa = cloudpickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename}.pkl not found.")
        except Exception:
            raise BaseException(f"File {filename}.pkl could not be loaded.")
        return msa

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
        alg_1_label = ""
        alg_2_label = ""
        _k_svm_scores = []
        _knn_scores = []
        subsets = os.listdir(self.dataset_path)

        for idx in range(len(subsets)):
            subset = f"{idx}_subset"
            subset_k_svm_scores = []
            subset_knn_scores = []
            feature_vectors = []
            actual_labels = []
            for idx, algorithm in enumerate(os.listdir(os.path.join(self.dataset_path, subset))):
                if idx == 0 and alg_1_label == "":
                    alg_1_label = algorithm
                elif idx == 1 and alg_2_label == "":
                    alg_2_label = algorithm

                for problem in os.listdir(os.path.join(self.dataset_path, subset, algorithm)):
                    runs = os.listdir(os.path.join(self.dataset_path, subset, algorithm, problem))
                    runs.sort()
                    for run in runs:
                        run_path = os.path.join(self.dataset_path, subset, algorithm, problem, run)
                        srd = SingleRunData.import_from_json(run_path)
                        feature_vector = srd.get_feature_vector(standard_scale=True)
                        feature_vectors.append(feature_vector)
                        actual_labels.append(idx)

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
                subset_k_svm_scores.append(tmp)

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
                subset_knn_scores.append(tmp)

            _k_svm_scores.append(np.mean(subset_k_svm_scores, axis=0))
            _knn_scores.append(np.mean(subset_knn_scores, axis=0))

        k_svm_scores = np.array(_k_svm_scores)
        knn_scores = np.array(_knn_scores)

        bar_width = 0.35
        (
            fig,
            ax,
        ) = plt.subplots(2, 1, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.5)

        # bar charts
        if bar_chart_filename is not None:
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

    def export_results_to_latex(self, generate_pdf: bool = False):
        r"""Generate latex file containing MSA results in form of tables.
        Optionally also generate pdf file.

        Returns:
            generate_pdf (bool): Generates, .pdf file. Only .tex file is generated if false.
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

        doc.append(Section(f"Comparison of {self.target_alg_abbr} and {self.optimized_alg_abbr}"))
        doc.append(Subsection("Comparison of hyperparameters settings and similarity metrics"))

        # Table comparing hyperparameters settings and similarity metrics
        if len(self.similarity_metrics) != 0:
            hyperparameters_table = self.get_hyperparameters_and_similarity_metrics_latex_table()
            doc.append(hyperparameters_table)
        else:
            doc.append(
                "Not able to display the table because similarity metrics were not calculated."
            )

        doc.append(Subsection("Comparison of fitness statistics"))

        # Table comparing fitness statistics
        fitness_table = self.get_fitness_comparison_latex_table()
        doc.append(fitness_table)

        if generate_pdf:
            doc.generate_pdf(
                os.path.join(
                    self.archive_path,
                    f"{self.target_alg_abbr}_{self.optimized_alg_abbr}_MSA_results",
                ),
                clean_tex=False,
            )
        else:
            doc.generate_tex(
                os.path.join(
                    self.archive_path,
                    f"{self.target_alg_abbr}_{self.optimized_alg_abbr}_MSA_results",
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
            + " c" * len(self.meta_ga.gene_spaces[next(iter(self.meta_ga.gene_spaces))])
            + " |"
            + " c" * len(self.similarity_metrics)
        )
        table = LongTable(table_specs)
        mc_target = MultiColumn(
            len(self.target_gene_space[next(iter(self.target_gene_space))]),
            align="c|",
            data=self.target_alg_abbr,
        )
        mc_optimized = MultiColumn(
            len(self.meta_ga.gene_spaces[next(iter(self.meta_ga.gene_spaces))]),
            align="c|",
            data=self.optimized_alg_abbr,
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
        for alg_name in self.meta_ga.gene_spaces:
            for setting in self.meta_ga.gene_spaces[alg_name]:
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
        for idx, (target, optimized, smape, cosine, rho, svm_test, knn_test) in enumerate(
            zip(self.target_solutions, self.optimized_solutions, *displayed_similarity_metrics)
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
            data=self.target_alg_abbr,
        )
        mc_optimized = MultiColumn(
            3,
            data=self.optimized_alg_abbr,
        )

        fitness_table.add_hline()
        fitness_table.add_row(["", mc_target, mc_optimized])

        fitness_table.add_hline()
        fitness_table.add_row(("c.n.", "min.", "mean.", "std.", "min.", "mean.", "std."))
        fitness_table.add_hline()

        subsets = os.listdir(self.dataset_path)

        # Collect fitness data from metaheuristic optimization runs
        fitness_statistics = []

        for alg_abbr in (self.target_alg_abbr, self.optimized_alg_abbr):
            mean_fitness = []
            min_fitness = []
            std_fitness = []
            for idx in range(len(subsets)):
                subset = f"{idx}_subset"
                runs = os.listdir(
                    os.path.join(
                        self.dataset_path,
                        subset,
                        alg_abbr,
                        self.meta_ga.problem.name(),
                    )
                )

                runs.sort()
                fitness = []

                for run in runs:
                    run_path = os.path.join(
                        self.dataset_path,
                        subset,
                        alg_abbr,
                        self.meta_ga.problem.name(),
                        run,
                    )

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
        r"""Produces a scheme of metaheuristic similarity analyzer configuration.

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
                    label="random set \nof target \nparameter settings",
                )

        with gv.subgraph(name="cluster_1") as c:
            c.attr(
                style="filled",
                color=graph_color,
                name="meta_ga",
                label="Meta GA",
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
                        <td>{self.meta_ga.rng_seed}</td>
                    </tr>
                </table>>"""

            c.node_attr.update(
                style="filled",
                color=table_border_color,
                fillcolor=table_background_color,
                shape="box",
            )

            c.node(
                name="cosine_similarity",
                label="Cosine similarity",
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
                for alg_idx, alg_name in enumerate(self.meta_ga.gene_spaces):
                    algorithm = get_algorithm_by_name(alg_name)
                    node_label = f"""<<table border="0" cellborder="1" cellspacing="0">
                        <tr>
                            <td colspan="2"><b>{algorithm.Name[1]}</b></td>
                        </tr>
                        <tr>
                            <td>pop size</td>
                            <td>{self.meta_ga.pop_size}</td>
                        </tr>"""
                    for setting in self.meta_ga.gene_spaces[alg_name]:
                        gene = ", ".join(str(value) for value in self.meta_ga.gene_spaces[alg_name][setting].values())
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

                for alg_idx in range(len(self.meta_ga.gene_spaces)):
                    cc.edge(f"gene_space_{alg_idx}", "combined_gene_space")

            c.edge("cosine_similarity", "combined_gene_space:gene_fitness")

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
            label="for every set of \ntarget parameters",
            lhead="cluster_2",
        )
        gv.edge(
            tail_name="pop_metrics",
            head_name="cosine_similarity",
            label="average feature vectors \nof target and optimized \nalgorithms \ndiversity metrics",
            ltail="cluster_2",
        )

        gv.attr(fontsize="25")

        gv.render(format="png", cleanup=True)
