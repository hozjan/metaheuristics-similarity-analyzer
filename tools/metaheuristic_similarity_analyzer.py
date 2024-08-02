from datetime import datetime
from pathlib import Path
import os
from typing import Any
from util.random import random_float_with_step
from tools.meta_ga import MetaGA, MetaGAFitnessFunction
from tools.optimization_tools import optimization_runner
from util.optimization_data import SingleRunData
import numpy as np
from scipy import spatial
import graphviz
import cloudpickle
from niapy.algorithms import Algorithm
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
        steps = []
        for alg_name in self.target_gene_space:
            if not isinstance(alg_name, str) and issubclass(alg_name, Algorithm):
                algorithm = alg_name()
            elif alg_name not in _algorithm_options():
                raise KeyError(
                    f"Could not find algorithm by name `{alg_name}` in the niapy library."
                )
            else:
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
                steps.append(self.target_gene_space[alg_name][setting]["step"])

        for _ in range(self.comparisons):
            target_solution = []
            for low, high, step in zip(low_ranges, high_ranges, steps):
                target_solution.append(random_float_with_step(low=low, high=high, step=step))
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

    def run_similarity_analysis(self, get_info=False, generate_dataset=False):
        r"""Run metaheuristic similarity analysis.

        Args:
            get_info (Optional[bool]): Generate info scheme of the metaheuristic similarity analyzer (false by default).
            generate_dataset (Optional[bool]): Generate dataset from target and optimized solutions after analysis (false by default).
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

        with gv.subgraph(name="cluster_0") as c:
            c.attr(style="filled", color=graph_color, name="msa", label="MSA")
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

                for algorithm in list(self.target_gene_space.keys()):
                    if not isinstance(algorithm, str) and issubclass(
                        algorithm, Algorithm
                    ):
                        alg_name = algorithm.Name[0]
                    elif isinstance(algorithm, str):
                        alg_name = algorithm

                    node_label = f"""<<table border="0" cellborder="1" cellspacing="0">
                        <tr>
                            <td colspan="2"><b>{alg_name}</b></td>
                        </tr>
                        <tr>
                            <td>pop size</td>
                            <td>{self.meta_ga.pop_size}</td>
                        </tr>"""
                    for setting in self.target_gene_space[algorithm]:
                        gene = ", ".join(
                            str(value)
                            for value in self.target_gene_space[algorithm][
                                setting
                            ].values()
                        )
                        node_label += f"<tr><td>{setting}</td><td>[{gene}]</td></tr>"
                    node_label += "</table>>"
                    cc.node(name=f"target_gene_space", label=node_label)

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
                for alg_idx, algorithm in enumerate(
                    list(self.meta_ga.gene_spaces.keys())
                ):
                    if not isinstance(algorithm, str) and issubclass(
                        algorithm, Algorithm
                    ):
                        alg_name = algorithm.Name[0]
                    elif isinstance(algorithm, str):
                        alg_name = algorithm

                    node_label = f"""<<table border="0" cellborder="1" cellspacing="0">
                        <tr>
                            <td colspan="2"><b>{alg_name}</b></td>
                        </tr>
                        <tr>
                            <td>pop size</td>
                            <td>{self.meta_ga.pop_size}</td>
                        </tr>"""
                    for setting in self.meta_ga.gene_spaces[algorithm]:
                        gene = ", ".join(
                            str(value)
                            for value in self.meta_ga.gene_spaces[algorithm][
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

                for alg_idx in range(len(list(self.meta_ga.gene_spaces.keys()))):
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
                        <td>max iters</td>
                        <td>{self.meta_ga.max_iters}</td>
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
            tail_name="combined_gene_space",
            head_name="pop_metrics",
            label=f"for each solution",
            lhead="cluster_2",
        )
        gv.edge(
            tail_name="optimization_parameters",
            head_name="pop_scheme",
            ltail="cluster_2",
            lhead="cluster_3",
        )
        gv.edge(
            tail_name="optimization_parameters",
            head_name="cosine_similarity",
            label="average feature vector \nof the optimized algorithm",
            ltail="cluster_2",
        )
        gv.edge(
            tail_name="target_gene_space",
            head_name="cosine_similarity",
            label="average feature vector \nof the target algorithm",
        )

        gv.attr(fontsize="25")

        gv.render(format="png", cleanup=True)
