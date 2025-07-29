import numpy.typing as npt
from niapy.problems import Problem
from niapy.algorithms import Algorithm
from datetime import datetime
from pathlib import Path
import numpy as np
import pygad
from msa.tools.optimization_tools import optimization_runner
from msa.tools.optimization_data import SingleRunData, IndivDiversityMetric, PopDiversityMetric
from msa.util.helper import get_algorithm_by_name
import shutil
import logging
import graphviz
from enum import Enum
import os


class MetaGAFitnessFunction(Enum):
    PARAMETER_TUNING = 0
    TARGET_PERFORMANCE_SIMILARITY = 1


__all__ = ["MetaGA"]


class MetaGA:
    r"""Class containing metadata of meta genetic algorithm."""

    # TODO add attributes

    def __init__(
        self,
        fitness_function_type: MetaGAFitnessFunction,
        ga_generations: int,
        ga_solutions_per_pop: int,
        ga_percent_parents_mating: int,
        ga_parent_selection_type: str,
        ga_k_tournament: int,
        ga_crossover_type: str,
        ga_mutation_type: str,
        ga_crossover_probability: float,
        ga_mutation_num_genes: int,
        ga_keep_elitism: int,
        gene_spaces: dict[str | Algorithm, dict[str, dict[str, float]]],
        pop_size: int,
        problem: Problem,
        max_iters: int | float = np.inf,
        max_evals: int | float = np.inf,
        num_runs: int = 10,
        pop_diversity_metrics: list[PopDiversityMetric] | None = None,
        indiv_diversity_metrics: list[IndivDiversityMetric] | None = None,
        rng_seed: int | None = None,
        base_archive_path="archive",
    ):
        r"""Initialize meta genetic algorithm.

        Args:
            fitness_function_type (MetaGAFitnessFunction): Type of fitness function of meta genetic algorithm.
            ga_generations (int): Number of generations of the genetic algorithm.
            ga_solutions_per_pop (int): Number of solutions per generation of the genetic algorithm.
            ga_percent_parents_mating (int): Percentage of parents mating for production of the offspring of the
                genetic algorithm [1, 100].
            ga_parent_selection_type (str): Type of parent selection of the genetic algorithm.
            ga_k_tournament (int): Number of parents participating in the tournament selection of the genetic
                algorithm. Only has effect when ga_parent_selection_type equals 'tournament'.
            ga_crossover_type (str): Crossover type of the genetic algorithm.
            ga_mutation_type (str): Mutation type of the genetic algorithm.
            ga_crossover_probability (float): Crossover probability of the genetic algorithm [0,1].
            ga_mutation_num_genes (int): Number of genes mutated in the solution of the genetic algorithm.
            ga_keep_elitism (int): Number of solutions that are a part of the elitism of the genetic algorithm.
            gene_spaces (dict[str | Algorithm, dict[str, dict[str, float]]]): Gene spaces of the solution.
            pop_size (int): Population size of the metaheuristics being optimized.
            problem (Problem): Optimization problem used for optimization.
            max_iters (Optional[int | float]): Maximum number of iterations of the metaheuristic being optimized for
                each solution of the genetic algorithm.
            max_evals (Optional[int | float]): Maximum number of evaluations of the metaheuristic being optimized for
                each solution of the genetic algorithm.
            num_runs (Optional[int]): Number of runs performed by the metaheuristic being optimized for each solution
                of the genetic algorithm.
            pop_diversity_metrics (Optional[list[PopDiversityMetric]]): List of population diversity metrics calculated.
                Only has effect when fitness_function_type set to `TARGET_PERFORMANCE_SIMILARITY`.
            indiv_diversity_metrics (Optional[list[IndivDiversityMetric]]): List of individual diversity metrics
                calculated. Only has effect when fitness_function_type set to `TARGET_PERFORMANCE_SIMILARITY`.
            rng_seed (Optional[int]): Seed of the random generator. Provide for reproducible results. Only has effect
                when fitness_function_type set to `TARGET_PERFORMANCE_SIMILARITY`.
            base_archive_path (Optional[str]): Base archive path of the meta genetic algorithm.

            ValueError: No diversity metrics defined when required.
            ValueError: Neither of `max_evals` or `max_iters` was assigned a finite value.
        """

        if fitness_function_type is MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY and (
            pop_diversity_metrics is None and indiv_diversity_metrics is None
        ):
            raise ValueError(
                "Diversity metrics must be defined when fitness_function_type set to `PERFORMANCE_SIMILARITY`."
            )

        if max_evals == np.inf and max_iters == np.inf:
            raise ValueError("Defining a finite value for max_evals and/or max_iters is required.")

        self.fitness_function_type = fitness_function_type
        self.gene_spaces = gene_spaces
        self.problem = problem
        self.ga_generations = ga_generations
        self.ga_solutions_per_pop = ga_solutions_per_pop
        self.ga_percent_parents_mating = ga_percent_parents_mating
        self.ga_parent_selection_type = ga_parent_selection_type
        self.ga_k_tournament = ga_k_tournament
        self.ga_crossover_type = ga_crossover_type
        self.ga_mutation_type = ga_mutation_type
        self.ga_crossover_probability = ga_crossover_probability
        self.ga_mutation_num_genes = ga_mutation_num_genes
        self.ga_keep_elitism = ga_keep_elitism
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.max_evals = max_evals
        self.num_runs = num_runs
        self.pop_diversity_metrics = pop_diversity_metrics
        self.indiv_diversity_metrics = indiv_diversity_metrics
        self.rng_seed = rng_seed
        self.base_archive_path = base_archive_path

        self.meta_ga: pygad.GA | None = None
        self.combined_gene_space: list[dict[str, float]] = []
        self.low_ranges: list[float] = []
        self.high_ranges: list[float] = []
        self.random_mutation_min_val: list[float] = []
        self.random_mutation_max_val: list[float] = []
        self.archive_path = ""
        self.__fitness_function = None
        self.__algorithms: list[str] = []
        self.__target_algorithm: Algorithm | None = None
        self.__meta_dataset = "meta_dataset"
        self.__meta_ga_tmp_data_path = ""
        self.__init_parameters()

    def __init_parameters(self):
        """
        Initialize meta genetic algorithm parameters and create folder structure.

        Raises:
            ValueError: Algorithm does not have the attribute provided in the `gene_spaces`.
            ValueError: Incorrect number of algorithms provided in the `gene_spaces`.
            ValueError: Incorrect number of algorithms provided in the `gene_spaces`.
            ValueError: Incorrect number of algorithms provided in the `gene_spaces`.
            TypeError: Provided `problem` is of incorrect type.
        """
        # check if all values in the provided gene spaces are correct and
        # assemble combined gene space for meta GA
        for alg_name in self.gene_spaces:
            algorithm = get_algorithm_by_name(alg_name)
            self.__algorithms.append(algorithm.Name[1])
            for setting in self.gene_spaces[alg_name]:
                if not hasattr(algorithm, setting):
                    raise ValueError(f"Algorithm `{alg_name}` has no attribute named `{setting}`.")
                self.combined_gene_space.append(self.gene_spaces[alg_name][setting])
                self.low_ranges.append(self.gene_spaces[alg_name][setting]["low"])
                self.high_ranges.append(self.gene_spaces[alg_name][setting]["high"])
                self.random_mutation_max_val.append(
                    abs(self.gene_spaces[alg_name][setting]["high"] - self.gene_spaces[alg_name][setting]["low"]) * 0.5
                )
                self.random_mutation_min_val.append(-self.random_mutation_max_val[-1])

        if self.fitness_function_type == MetaGAFitnessFunction.PARAMETER_TUNING:
            if len(self.__algorithms) != 1:
                raise ValueError(
                    """Only one algorithm must be defined in the `gene_spaces` provided
                    when fitness_function_type set to `PARAMETER_TUNING`."""
                )
            self.__fitness_function = self.meta_ga_fitness_function_for_parameter_tuning
        elif self.fitness_function_type == MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY:
            if len(self.__algorithms) != 1:
                raise ValueError(
                    """Only one algorithm must be defined in `gene_spaces` provided when
                    fitness_function_type set to `TARGET_PERFORMANCE_SIMILARITY`."""
                )
            self.__fitness_function = self.meta_ga_fitness_function_for_target_performance_similarity

        # check if the provided optimization problem is correct
        if not isinstance(self.problem, Problem):
            raise TypeError(f"Provided problem type `{type(self.problem).__name__}` is not compatible.")

    def __create_folder_structure(self, prefix: str | None = None):
        r"""Create folder structure for the meta genetic algorithm.

        Args:
            prefix (Optional[str]): Use custom prefix for the name of the base
                folder in structure. Uses current datetime by default.
        """
        if prefix is None:
            prefix = str(datetime.now().strftime("%m-%d_%H.%M.%S"))
        self.archive_path = os.path.join(
            self.base_archive_path,
            "_".join([prefix, *self.__algorithms, self.problem.name()]),
        )
        self.__meta_ga_tmp_data_path = os.path.join(self.archive_path, "meta_ga_tmp_data")
        if os.path.exists(self.archive_path) is False:
            Path(self.archive_path).mkdir(parents=True, exist_ok=True)

    def __get_logger(self, filename):
        r"""Get logger for meta genetic algorithm. Outputs to file and console.

        Args:
            filename (str): Log file name.

        Returns:
            logger (Logger)
        """
        level = logging.DEBUG

        logger = logging.getLogger("meta_ga_logger")
        logger.setLevel(level)
        logger.propagate = False

        file_handler = logging.FileHandler(f"{filename}.txt", "a+", "utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def solution_to_algorithm_attributes(
        solution: list[float] | npt.NDArray,
        gene_spaces: dict[str | Algorithm, dict[str, dict[str, float]]],
        pop_size: int,
    ):
        r"""Apply meta genetic algorithm solution to an corresponding algorithms based on
        the gene spaces used for the meta optimization. Make sure the solution matches the gene space.

        Args:
            solution (list[float | numpy.ndarray]): Meta genetic algorithm solution.
            gene_spaces (dict[str | Algorithm, dict[str, dict[str, float]]]): Gene spaces of the solution.
            pop_size (int): Population size of the algorithms returned.

        Returns:
            list: Array of Algorithms configured based on solution and gene_space.

        Raises:
            ValueError: The length of the provided solution does not match the number of attributes in `gene_spaces`.
            ValueError: Algorithm does not have the attribute provided in the `gene_spaces`.
        """
        settings_counter = 0
        for alg_name in gene_spaces:
            settings_counter += len(gene_spaces[alg_name])
        if settings_counter != len(solution):
            raise ValueError("Solution length does not match the count of the gene space settings.")

        solution_iter = 0
        algorithms = []
        for alg_name in gene_spaces:
            algorithm = get_algorithm_by_name(alg_name, population_size=pop_size)
            for setting in gene_spaces[alg_name]:
                if not hasattr(algorithm, setting):
                    raise ValueError(f"Algorithm `{alg_name}` has no attribute named `{setting}`.")
                algorithm.__setattr__(setting, solution[solution_iter])
                solution_iter += 1
            algorithms.append(algorithm)
        return algorithms

    @staticmethod
    def on_generation_progress(ga: pygad.GA):
        r"""Called after each genetic algorithm generation."""
        ga.logger.info(f"Generation = {ga.generations_completed}")
        ga.logger.info(f"->Fitness  = {ga.best_solution(pop_fitness=ga.last_generation_fitness)[1]}")
        ga.logger.info(f"->Solution = {ga.best_solution(pop_fitness=ga.last_generation_fitness)[0]}")

    def __clean_tmp_data(self):
        r"""Clean up temporary data created by the meta genetic algorithm."""
        try:
            print("Cleaning up meta GA temporary data...")
            if os.path.exists(self.__meta_ga_tmp_data_path):
                shutil.rmtree(self.__meta_ga_tmp_data_path)
        except Exception:
            print("Cleanup failed!")

    def meta_ga_fitness_function_for_parameter_tuning(self, meta_ga, solution, solution_idx):
        r"""Fitness function of the meta genetic algorithm.
        For tuning parameters of metaheuristic algorithms for best performance."""

        meta_dataset = os.path.join(self.__meta_ga_tmp_data_path, f"{solution_idx}_{self.__meta_dataset}")

        algorithms = MetaGA.solution_to_algorithm_attributes(solution, self.gene_spaces, self.pop_size)

        # gather optimization data
        for algorithm in algorithms:
            optimization_runner(
                algorithm=algorithm,
                problem=self.problem,
                runs=self.num_runs,
                dataset_path=meta_dataset,
                max_iters=self.max_iters,
                max_evals=self.max_evals,
                run_index_seed=True,
                keep_pop_data=False,
                keep_diversity_metrics=False,
                parallel_processing=True,
            )

        fitness_values = []
        for algorithm in os.listdir(meta_dataset):
            for problem in os.listdir(os.path.join(meta_dataset, algorithm)):
                runs = os.listdir(os.path.join(meta_dataset, algorithm, problem))
                for run in runs:
                    run_path = os.path.join(meta_dataset, algorithm, problem, run)
                    fitness_values.append(SingleRunData.import_from_json(run_path).best_fitness)

        avg_fitness = np.average(fitness_values)

        return 1.0 / avg_fitness + 0.0000000001

    def meta_ga_fitness_function_for_target_performance_similarity(self, meta_ga, solution, solution_idx):
        r"""Fitness function of the meta genetic algorithm.
        For tuning parameters of metaheuristic algorithm for best similarity of diversity metrics.
        """

        meta_dataset = os.path.join(self.__meta_ga_tmp_data_path, f"{solution_idx}_{self.__meta_dataset}")

        algorithms = MetaGA.solution_to_algorithm_attributes(solution, self.gene_spaces, self.pop_size)

        # gather optimization data
        for algorithm in algorithms:
            optimization_runner(
                algorithm=algorithm,
                problem=self.problem,
                runs=self.num_runs,
                dataset_path=meta_dataset,
                pop_diversity_metrics=self.pop_diversity_metrics,
                indiv_diversity_metrics=self.indiv_diversity_metrics,
                max_iters=self.max_iters,
                max_evals=self.max_evals,
                run_index_seed=True,
                keep_pop_data=False,
                parallel_processing=True,
            )

        target_runs_path = os.path.join(
            self.__meta_ga_tmp_data_path,
            self.__target_algorithm.Name[1],
            self.problem.name(),
        )

        target_runs = os.listdir(target_runs_path)
        target_runs.sort()

        optimized_runs = []
        optimized_runs_path = ""
        for algorithm in os.listdir(meta_dataset):
            for problem in os.listdir(os.path.join(meta_dataset, algorithm)):
                optimized_runs_path = os.path.join(meta_dataset, algorithm, problem)
                optimized_runs = os.listdir(optimized_runs_path)
                optimized_runs.sort()

        similarities = []
        for target, optimized in zip(target_runs, optimized_runs):
            target_srd = SingleRunData.import_from_json(os.path.join(target_runs_path, target))
            optimized_srd = SingleRunData.import_from_json(os.path.join(optimized_runs_path, optimized))
            similarities.append(target_srd.get_diversity_metrics_similarity(optimized_srd))

        return np.mean(similarities)

    def __before_meta_optimization(self):
        r"""Execute before meta genetic algorithm optimization."""
        if (
            self.fitness_function_type == MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY
            and self.__target_algorithm is not None
        ):
            optimization_runner(
                algorithm=self.__target_algorithm,
                problem=self.problem,
                runs=self.num_runs,
                dataset_path=self.__meta_ga_tmp_data_path,
                pop_diversity_metrics=self.pop_diversity_metrics,
                indiv_diversity_metrics=self.indiv_diversity_metrics,
                max_iters=self.max_iters,
                max_evals=self.max_evals,
                run_index_seed=True,
                keep_pop_data=False,
                parallel_processing=True,
            )

    def run_meta_ga(
        self,
        filename="meta_ga_obj",
        plot_filename="meta_ga_fitness_plot",
        log_filename="meta_ga_log_file",
        target_algorithm: Algorithm | None = None,
        get_info: bool = False,
        prefix: str | None = None,
        return_best_solution: bool = False,
    ):
        r"""Run meta genetic algorithm. Saves pygad.GA instance and fitness plot image as a result of the optimization.

        Args:
            filename (Optional[str]): Name of the .pkl file of the GA object created during optimization.
            plot_filename (Optional[str]): Name of the fitness plot image file.
            log_filename (Optional[str]): Name of the generated log file.
            target_algorithm (Optional[Algorithm]): Target algorithm for the performance similarity evaluation.
                Only required when fitness_function_type set to `TARGET_PERFORMANCE_SIMILARITY`.
            get_info (Optional[bool]): Generate info scheme of the meta genetic algorithm (false by default).
            prefix (Optional[str]): Use custom prefix for the name of the base folder in structure. Uses current
                datetime by default.
            return_best_solution (Optional[bool]): returns best solution if True.

        Returns:
            best_solution (numpy.ndarray[float] | None): Returns best solution.

        Raises:
            ValueError: `target_algorithm` was not provided when required.
        """
        if (
            self.fitness_function_type == MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY
            and target_algorithm is None
        ):
            raise ValueError(
                """target_algorithm must be defined when running optimization with
                fitness_function_type set to `TARGET_PERFORMANCE_SIMILARITY`."""
            )
        else:
            self.__target_algorithm = target_algorithm

        self.__create_folder_structure(prefix=prefix)
        self.__before_meta_optimization()

        if get_info:
            self.meta_ga_info(
                filename=os.path.join(self.archive_path, "meta_ga_info"),
            )

        num_parents_mating = int(self.ga_solutions_per_pop * (self.ga_percent_parents_mating / 100))

        self.meta_ga = pygad.GA(
            num_generations=self.ga_generations,
            num_parents_mating=num_parents_mating,
            keep_elitism=self.ga_keep_elitism,
            allow_duplicate_genes=False,
            fitness_func=self.__fitness_function,
            sol_per_pop=self.ga_solutions_per_pop,
            num_genes=len(self.combined_gene_space),
            parent_selection_type=self.ga_parent_selection_type,
            K_tournament=self.ga_k_tournament,
            init_range_low=self.low_ranges,
            init_range_high=self.high_ranges,
            crossover_type=self.ga_crossover_type,
            crossover_probability=self.ga_crossover_probability,
            mutation_type=self.ga_mutation_type,
            mutation_num_genes=self.ga_mutation_num_genes,
            random_mutation_min_val=self.random_mutation_min_val,
            random_mutation_max_val=self.random_mutation_max_val,
            gene_space=self.combined_gene_space,
            on_generation=self.on_generation_progress,
            save_best_solutions=True,
            save_solutions=True,
            logger=self.__get_logger(filename=os.path.join(self.archive_path, log_filename)),
        )

        self.meta_ga.run()
        self.__clean_tmp_data()

        self.meta_ga.logger.handlers.clear()
        self.meta_ga.save(os.path.join(self.archive_path, filename))
        self.meta_ga.plot_fitness(save_dir=os.path.join(self.archive_path, f"{plot_filename}.png"))
        best_solutions = self.meta_ga.best_solutions
        print(f"Best solution: {best_solutions[-1]}")

        if return_best_solution:
            return np.array(best_solutions[-1])

    def meta_ga_info(
        self,
        filename: str = "meta_ga_info",
        table_background_color: str = "white",
        table_border_color: str = "black",
        graph_color: str = "grey",
        sub_graph_color: str = "lightgrey",
    ):
        r"""Produces a scheme of meta genetic algorithm configuration.

        Args:
            filename (Optional[str]): Name of the scheme image file.
            table_background_color (Optional[str]): Table background color.
            table_border_color (Optional[str]): Table border color.
            graph_color (Optional[str]): Graph background color.
            sub_graph_color (Optional[str]): Sub graph background color.
        """

        gv = graphviz.Digraph("meta_ga_info", filename=filename)
        gv.attr(rankdir="TD", compound="true")
        gv.attr("node", shape="box")
        gv.attr("graph", fontname="bold")

        with gv.subgraph(name="cluster_0") as c:
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
                        <td>{self.ga_generations}</td>
                    </tr>
                    <tr>
                        <td>pop size</td>
                        <td>{self.ga_solutions_per_pop}</td>
                    </tr>
                    <tr>
                        <td>parent selection</td>
                        <td>{self.ga_parent_selection_type}</td>
                    </tr>
                    """

            if self.ga_parent_selection_type == "tournament":
                meta_ga_parameters_label += f"""
                    <tr>
                        <td>K tournament</td>
                        <td>{self.ga_k_tournament}</td>
                    </tr>"""
            meta_ga_parameters_label += f"""
                    <tr>
                        <td>parents</td>
                        <td>{self.ga_percent_parents_mating} %</td>
                    </tr>
                    <tr>
                        <td>crossover type</td>
                        <td>{self.ga_crossover_type}</td>
                    </tr>
                    <tr>
                        <td>mutation type</td>
                        <td>{self.ga_mutation_type}</td>
                    </tr>
                    <tr>
                        <td>crossover prob.</td>
                        <td>{self.ga_crossover_probability}</td>
                    </tr>
                    <tr>
                        <td>mutate num genes</td>
                        <td>{self.ga_mutation_num_genes}</td>
                    </tr>
                    <tr>
                        <td>keep elitism</td>
                        <td>{self.ga_keep_elitism}</td>
                    </tr>
                    <tr>
                        <td>rng seed</td>
                        <td>{self.rng_seed}</td>
                    </tr>
                </table>>"""
            c.node(name="meta_ga_parameters", label=meta_ga_parameters_label)

            with c.subgraph(name="cluster_00") as cc:
                cc.attr(
                    style="filled",
                    color=sub_graph_color,
                    name="meta_ga_algorithms",
                    label="Algorithms",
                )
                cc.node_attr.update(
                    style="filled",
                    color=table_border_color,
                    fillcolor=table_background_color,
                    shape="plaintext",
                    margin="0",
                )
                combined_gene_space_len = 0
                for alg_idx, alg_name in enumerate(list(self.gene_spaces)):
                    algorithm = get_algorithm_by_name(alg_name)

                    node_label = f"""<<table border="0" cellborder="1" cellspacing="0">
                        <tr>
                            <td colspan="2"><b>{algorithm.Name[1]}</b></td>
                        </tr>
                        <tr>
                            <td>pop size</td>
                            <td>{self.pop_size}</td>
                        </tr>"""
                    for setting in self.gene_spaces[alg_name]:
                        gene = ", ".join(str(value) for value in self.gene_spaces[alg_name][setting].values())
                        combined_gene_space_len += 1
                        node_label += f"""<tr>
                            <td>{setting}</td><td>[{gene}]<sub> g<i>{combined_gene_space_len}</i></sub></td>
                        </tr>"""
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

                for alg_idx in range(len(self.gene_spaces)):
                    cc.edge(f"gene_space_{alg_idx}", "combined_gene_space")

        with gv.subgraph(name="cluster_1") as c:
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
                        <td>{self.max_iters}</td>
                    </tr>
                    <tr>
                        <td>max evals</td>
                        <td>{self.max_evals}</td>
                    </tr>
                    <tr>
                        <td>num runs</td>
                        <td>{self.num_runs}</td>
                    </tr>
                    <tr>
                        <td>problem</td>
                        <td>{self.problem.name()}</td>
                    </tr>
                    <tr>
                        <td>dimension</td>
                        <td>{self.problem.dimension}</td>
                    </tr>
                </table>>""",
            )

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
                                    <td><i><b>X</b><sub>i=1, t={self.max_iters}</sub></i></td>
                                </tr>
                                <tr>
                                    <td>...</td>
                                    <td>...</td>
                                    <td>...</td>
                                </tr>
                                <tr>
                                    <td><i><b>X</b><sub>i={self.pop_size}, t=1</sub></i></td>
                                    <td>...</td>
                                    <td><i><b>X</b><sub>i={self.pop_size}, t={self.max_iters}</sub></i></td>
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
                                    <td><i><b>IM</b><sub>{self.pop_size}</sub></i></td>
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
                                    <td><i><b>PM</b><sub>{self.max_iters}</sub></i></td>
                                </tr>
                            </table>
                        </td>
                        <td></td>
                    </tr>
                </table>>""",
            )

        gv.edge(
            tail_name="combined_gene_space",
            head_name="optimization_parameters",
            label="for each \nsolution",
            lhead="cluster_1",
        )
        gv.edge(
            tail_name="optimization_parameters",
            head_name="pop_scheme",
            ltail="cluster_1",
            lhead="cluster_3",
        )
        gv.edge(
            tail_name="optimization_parameters",
            head_name="combined_gene_space:gene_fitness",
            label=(
                "average fitness \nof runs"
                if self.fitness_function_type == MetaGAFitnessFunction.PARAMETER_TUNING
                else "cosine distance \nof average feature vectors\nof target and optimized algorithm"
            ),
            ltail="cluster_1",
        )

        gv.attr(fontsize="25")

        gv.render(format="png", cleanup=True)
