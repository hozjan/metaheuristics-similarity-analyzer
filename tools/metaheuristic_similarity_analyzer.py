from datetime import datetime
from logging import warning
from pathlib import Path
import warnings
import os
from typing import Any
from pylatex import Document, Section, Subsection
from pylatex import MultiColumn, Package, LongTable
from pylatex.utils import bold
from torch import Value
from util.helper import random_float_with_step, get_algorithm_by_name
from tools.meta_ga import MetaGA, MetaGAFitnessFunction
from tools.optimization_tools import optimization_runner
from tools.ml_tools import svm_and_knn_classification
from util.optimization_data import SingleRunData
import numpy as np
from scipy import spatial, stats
import graphviz
import cloudpickle
from niapy.algorithms import Algorithm
from niapy.problems import Problem
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
        target_gene_space: dict[str | Algorithm, dict[str, Any]] = None,
        comparisons: int = 0,
    ) -> None:
        r"""Initialize the metaheuristic similarity analyzer.

        Args:
            meta_ga (Optional[MetaGA]): Preconfigured instance of the meta genetic algorithm with fitness function set to `TARGET_PERFORMANCE_SIMILARITY`.
            target_gene_space (Optional[dict[str | Algorithm, dict[str, Any]]]): Gene space of the reference metaheuristic.
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

        if len(target_gene_space) != 1:
            raise ValueError(
                "Only one algorithm must be defined in `target_gene_space` provided."
            )

        self.meta_ga = meta_ga
        self.target_gene_space = target_gene_space
        self.comparisons = comparisons
        self.target_solutions = []
        self.optimized_solutions = []
        self.similarity_metrics = {}
        self.archive_path = ""
        self.dataset_path = ""
        self.target_alg_abbr = get_algorithm_by_name(list(target_gene_space)[0]).Name[1]
        self.optimized_alg_abbr = get_algorithm_by_name(
            list(self.meta_ga.gene_spaces)[0]
        ).Name[1]

    def __generate_targets(self, generate_optimized_targets: bool = False):
        r"""Generate target solutions.

        Args:
            generate_optimized_targets (Optional[bool]): Generate target solutions by parameter tuning if True, otherwise generate random targets.
        """
        low_ranges = []
        high_ranges = []
        steps = []

        for alg_name in self.target_gene_space:
            algorithm = get_algorithm_by_name(alg_name)
            for setting in self.target_gene_space[alg_name]:
                if not hasattr(algorithm, setting):
                    raise NameError(
                        f"Algorithm `{alg_name}` has no attribute named `{setting}`."
                    )
                low_ranges.append(self.target_gene_space[alg_name][setting]["low"])
                high_ranges.append(self.target_gene_space[alg_name][setting]["high"])
                steps.append(self.target_gene_space[alg_name][setting]["step"])

        self.__create_folder_structure()

        for idx in range(self.comparisons):
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
                target_solution = meta_ga.run_meta_ga(
                    prefix=str(idx), return_best_solution=True
                )
                self.target_solutions.append(target_solution)
            else:
                target_solution = []
                for low, high, step in zip(low_ranges, high_ranges, steps):
                    target_solution.append(
                        random_float_with_step(low=low, high=high, step=step)
                    )
                self.target_solutions.append(np.array(target_solution))

    def calculate_similarity_metrics(self):
        r"""Calculates similarity metrics from diversity metrics values of the comparisons stored in the generated dataset.
        If no dataset was created method will have no effect.
        """

        if os.path.exists(self.dataset_path) == False:
            raise (
                ValueError(
                    "Dataset does not exist. Run `generate_dataset_from_solutions` to generate dataset!"
                )
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

            first_runs = os.listdir(
                os.path.join(self.dataset_path, subset, self.target_alg_abbr, problem)
            )
            second_runs = os.listdir(
                os.path.join(
                    self.dataset_path, subset, self.optimized_alg_abbr, problem
                )
            )

            first_runs.sort()
            second_runs.sort()

            smape_values = []
            for fr, sr in zip(first_runs, second_runs):
                first_run_path = os.path.join(
                    self.dataset_path, subset, self.target_alg_abbr, problem, fr
                )
                second_run_path = os.path.join(
                    self.dataset_path, subset, self.optimized_alg_abbr, problem, sr
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
        ml_accuracy = svm_and_knn_classification(self.dataset_path, 100)

        self.similarity_metrics["smape"] = mean_smape
        self.similarity_metrics["cosine"] = cosine_similarity
        self.similarity_metrics["spearman_r"] = spearman_r
        self.similarity_metrics.update(ml_accuracy)

    def generate_dataset_from_solutions(self, num_runs: int = None):
        r"""Generate dataset from target and optimized solutions.

        Args:
            num_runs (Optional[int]): Number of runs performed by the metaheuristic for each solution. if None value assigned to meta genetic algorithm is used.
        """
        if num_runs is None:
            num_runs = self.meta_ga.num_runs

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
                    max_evals=self.meta_ga.max_evals,
                    run_index_seed=True,
                    keep_pop_data=False,
                    parallel_processing=True,
                )

    def __create_folder_structure(self):
        r"""Create folder structure for metaheuristic similarity analysis."""
        datetime_now = str(datetime.now().strftime("%m-%d_%H.%M.%S"))
        self.archive_path = os.path.join(
            "archive/target_performance_similarity",
            "_".join([datetime_now, self.target_alg_abbr, self.meta_ga.problem.name()]),
        )
        self.dataset_path = os.path.join(self.archive_path, "dataset")
        if os.path.exists(self.archive_path) == False:
            Path(self.archive_path).mkdir(parents=True, exist_ok=True)

    def run_similarity_analysis(
        self,
        target_solutions: dict[str, list[np.ndarray]] = None,
        generate_optimized_targets: bool = False,
        get_info=False,
        generate_dataset=False,
        calculate_similarity_metrics=False,
        export=False,
    ):
        r"""Run metaheuristic similarity analysis.

        Args:
            target_solutions (Optional[list[np.ndarray]]): Target solutions for the target algorithm. Length of the list must match `comparisons` parameter. Generated if None.
            generate_optimized_targets (Optional[bool]): Generate target solutions by parameter tuning. Target solutions wil be generated by uniform rng if false. Has no effect if `target_solutions` is not None.
            get_info (Optional[bool]): Generate info scheme of the metaheuristic similarity analyzer (false by default).
            generate_dataset (Optional[bool]): Generate dataset from target and optimized solutions after analysis (false by default).
            calculate_similarity_metrics (Optional[bool]): Calculates similarity metrics from target and optimized solutions after analysis (false by default). Has no effect if `generate_dataset` is false.
            export (Optional[bool]): Export MSA object to pkl after analysis.
        """
        if (
            self.meta_ga is None
            or self.meta_ga.fitness_function_type
            != MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY
        ):
            raise ValueError(
                "The `meta_ga` parameter must be defined and the fitness function must be set to `TARGET_PERFORMANCE_SIMILARITY`."
            )

        if target_solutions is None:
            self.__generate_targets(generate_optimized_targets)
        else:
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
            self.meta_ga.run_meta_ga(
                target_algorithm=target_algorithm[0], prefix=str(idx)
            )
            self.optimized_solutions.append(self.meta_ga.meta_ga.best_solutions[-1])

        if generate_dataset:
            self.generate_dataset_from_solutions()
            if calculate_similarity_metrics:
                print("Calculating similarity metrics...")
                self.calculate_similarity_metrics()
        if export:
            print("Exporting .pkl file...")
            self.export_to_pkl("msa_obj")

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
        """

        try:
            with open(filename + ".pkl", "rb") as file:
                msa = cloudpickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename}.pkl not found.")
        except:
            raise BaseException(f"File {filename}.pkl could not be loaded.")
        return msa

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

        doc.append(
            Section(
                f"Comparison of {self.target_alg_abbr} and {self.optimized_alg_abbr}"
            )
        )
        doc.append(Subsection("Comparison of hyperparameters settings"))

        # Table comparing hyperparameters settings
        hyperparameters_table = self.get_hyperparameters_latex_table()
        doc.append(hyperparameters_table)

        doc.append(Subsection("Comparison of similarity metrics"))

        # Table comparing similarity metrics
        if len(self.similarity_metrics) != 0:
            similarity_metrics_table = self.get_similarity_metrics_latex_table()
            doc.append(similarity_metrics_table)

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

    def get_hyperparameters_latex_table(self):
        r"""Create latex table displaying hyperparameters settings of target and optimized metaheuristic.

        Returns:
            hyperparameters_table (LongTable): Table displaying hyperparameters settings of both metaheuristics.
        """

        # Create table header
        table_specs = (
            "p{1cm} |"
            + " c" * len(self.target_gene_space.get(list(self.target_gene_space)[0]))
            + " |"
            + " c"
            * len(self.meta_ga.gene_spaces.get(list(self.meta_ga.gene_spaces)[0]))
        )
        hyperparameters_table = LongTable(table_specs)
        mc_target = MultiColumn(
            len(self.target_gene_space.get(list(self.target_gene_space)[0])),
            align="c|",
            data=self.target_alg_abbr,
        )
        mc_optimized = MultiColumn(
            len(self.meta_ga.gene_spaces.get(list(self.meta_ga.gene_spaces)[0])),
            data=self.optimized_alg_abbr,
        )

        hyperparameters_table.add_hline()
        hyperparameters_table.add_row(["", mc_target, mc_optimized])

        cells = ["c.n."]
        for alg_name in self.target_gene_space:
            for setting in self.target_gene_space[alg_name]:
                cells.append(setting)
        for alg_name in self.meta_ga.gene_spaces:
            for setting in self.meta_ga.gene_spaces[alg_name]:
                cells.append(setting)

        hyperparameters_table.add_hline()
        hyperparameters_table.add_row(cells)
        hyperparameters_table.add_hline()

        # Add rows
        for idx, (target, optimized) in enumerate(
            zip(self.target_solutions, self.optimized_solutions)
        ):
            cells = [f"{idx + 1}"]
            for t in target:
                cells.append(f"{round(t, 2)}")
            for o in optimized:
                cells.append(f"{round(o, 2)}")

            hyperparameters_table.add_row(cells)

        hyperparameters_table.add_hline()

        # Calculate statistics at the end of the table
        for stat in [np.min, np.mean, np.max, np.std]:
            hyperparameters_table.add_row(
                np.concatenate(
                    (
                        np.array([f"{stat.__name__}."]),
                        np.round(stat(np.array(self.target_solutions), axis=0), 2),
                        np.round(stat(np.array(self.optimized_solutions), axis=0), 2),
                    )
                )
            )

        hyperparameters_table.add_hline()
        return hyperparameters_table

    def get_similarity_metrics_latex_table(self):
        r"""Create latex table displaying similarity metrics.

        Returns:
            similarity_metrics_table (LongTable): Table displaying similarity metrics.
        """

        if len(self.similarity_metrics) == 0:
            raise ValueError(
                """Similarity metrics were not calculated and thus can not be displayed!
                    To calculate similarity metrics call method `calculate_similarity_metrics`"""
            )

        # Create table header
        similarity_metrics_table = LongTable(
            "*{1}{>{\centering\\arraybackslash}m{.05\paperwidth}} |"
            + " *{1}{>{\centering\\arraybackslash}m{.09\paperwidth}}"
            + " *{1}{>{\centering\\arraybackslash}m{.05\paperwidth}}"
            + " *{1}{>{\centering\\arraybackslash}m{.05\paperwidth}}"
            + " *{1}{>{\centering\\arraybackslash}m{.1\paperwidth}}"
            + " *{1}{>{\centering\\arraybackslash}m{.1\paperwidth}}"
        )
        similarity_metrics_table.add_hline()
        similarity_metrics_table.add_row(
            (
                "c.n.",
                " 1-SMAPE ",
                " cos.sim. ",
                " rho ",
                " 1-accuracy (SVM) ",
                " 1-accuracy (KNN)",
            )
        )
        similarity_metrics_table.add_hline()

        displayed_similarity_metrics = [
            self.similarity_metrics["smape"],
            self.similarity_metrics["cosine"],
            self.similarity_metrics["spearman_r"],
            self.similarity_metrics["svm_test"],
            self.similarity_metrics["knn_test"],
        ]

        # Add rows
        for idx, (smape, cosine, rho, svm_test, knn_test) in enumerate(
            zip(*displayed_similarity_metrics)
        ):
            row = [f"{idx + 1}"]
            for value, list in zip(
                [smape, cosine, rho, svm_test, knn_test],
                displayed_similarity_metrics,
            ):
                if round(value, 2) == round(np.max(list), 2):
                    row.append(bold(f"{round(value, 2)}"))
                else:
                    row.append(f"{round(value, 2)}")

            similarity_metrics_table.add_row(row)

        similarity_metrics_table.add_hline()

        # Calculate statistics at the end of the table
        for stat in (np.min, np.mean, np.max, np.std):
            row = [f"{stat.__name__}."]
            for list in displayed_similarity_metrics:
                row.append(round(stat(list), 2))
            similarity_metrics_table.add_row(row)

        similarity_metrics_table.add_hline()
        return similarity_metrics_table

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
        fitness_table.add_row(
            ("c.n.", "min.", "mean.", "std.", "min.", "mean.", "std.")
        )
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
        for idx, (min_1, mean_1, std_1, min_2, mean_2, std_2) in enumerate(
            zip(*fitness_statistics)
        ):
            row = [f"{idx + 1}"]
            for value, list in zip(
                [min_1, mean_1, std_1, min_2, mean_2, std_2], fitness_statistics
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
                        <td>{self.comparisons}</td>
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
                        gene = ", ".join(
                            str(value)
                            for value in self.target_gene_space[alg_name][
                                setting
                            ].values()
                        )
                        node_label += f"<tr><td>{setting}</td><td>[{gene}]</td></tr>"
                        target_parameters_len += 1
                    node_label += "</table>>"
                    cc.node(name=f"target_gene_space", label=node_label)

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
                cc.node(name=f"target_parameters", label=target_parameters)

                cc.edge(
                    "target_gene_space",
                    "target_parameters",
                    label="random set \nof target \nparameter settings",
                )

        with gv.subgraph(name="cluster_1") as c:
            c.attr(style="filled", color=graph_color, name="meta_ga", label="Meta GA")
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
                        gene = ", ".join(
                            str(value)
                            for value in self.meta_ga.gene_spaces[alg_name][
                                setting
                            ].values()
                        )
                        combined_gene_space_len += 1
                        node_label += f"<tr><td>{setting}</td><td>[{gene}]<sub> g<i>{combined_gene_space_len}</i></sub></td></tr>"
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
                cc.node(name=f"combined_gene_space", label=combined_gene_string)

                for alg_idx in range(len(self.meta_ga.gene_spaces)):
                    cc.edge(f"gene_space_{alg_idx}", "combined_gene_space")

            c.edge(f"cosine_similarity", "combined_gene_space:gene_fitness")

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
                pop_metrics_label = f'<<table border="0" cellborder="1" cellspacing="0"><tr><td><b>Pop Metrics</b></td></tr>'
                for metric in self.meta_ga.pop_diversity_metrics:
                    pop_metrics_label += f"""<tr><td>{metric.value}</td></tr>"""
                pop_metrics_label += "</table>>"
                cc.node(name=f"pop_metrics", label=pop_metrics_label)

                indiv_metrics_label = f'<<table border="0" cellborder="1" cellspacing="0"><tr><td><b>Indiv Metrics</b></td></tr>'
                for metric in self.meta_ga.indiv_diversity_metrics:
                    indiv_metrics_label += f"""<tr><td>{metric.value}</td></tr>"""
                indiv_metrics_label += "</table>>"
                cc.node(name=f"indiv_metrics", label=indiv_metrics_label)

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
                                    <td><i><b>X</b><sub>i=1, t={self.meta_ga.max_iters}</sub></i></td>
                                </tr>
                                <tr>
                                    <td>...</td>
                                    <td>...</td>
                                    <td>...</td>
                                </tr>
                                <tr>
                                    <td><i><b>X</b><sub>i={self.meta_ga.pop_size}, t=1</sub></i></td>
                                    <td>...</td>
                                    <td><i><b>X</b><sub>i={self.meta_ga.pop_size}, t={self.meta_ga.max_iters}</sub></i></td>
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
                                    <td><i><b>IM</b><sub>{self.meta_ga.pop_size}</sub></i></td>
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
                                    <td><i><b>PM</b><sub>{self.meta_ga.max_iters}</sub></i></td>
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
            xlabel=f"for each \nsolution",
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
