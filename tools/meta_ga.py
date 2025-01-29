from typing import Any
from niapy.util.factory import (
    _algorithm_options,
    get_algorithm,
)
from niapy.problems import Problem
from niapy.algorithms import Algorithm
from datetime import datetime
from pathlib import Path
from scipy import spatial
import torch
from torch import nn
import numpy as np
import pygad
import tools
from tools.optimization_tools import optimization_runner
from tools.ml_tools import get_data_loaders, nn_test, nn_train, LSTMClassifier
import tools.problems
from util.optimization_data import SingleRunData
from util.helper import get_algorithm_by_name
import shutil
import logging
import graphviz
from enum import Enum
import os
from util.pop_diversity_metrics import PopDiversityMetric
from util.indiv_diversity_metrics import IndivDiversityMetric


class MetaGAFitnessFunction(Enum):
    PARAMETER_TUNING = 0
    PERFORMANCE_SIMILARITY = 1
    TARGET_PERFORMANCE_SIMILARITY = 2


__all__ = ["MetaGA"]


class MetaGA:
    r"""Class containing metadata of meta genetic algorithm."""

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
        gene_spaces: dict[str | Algorithm, dict[str, Any]],
        pop_size: int,
        problem: Problem,
        max_iters: int = np.inf,
        max_evals: int = np.inf,
        num_runs: int = 10,
        pop_diversity_metrics: list[PopDiversityMetric] = None,
        indiv_diversity_metrics: list[IndivDiversityMetric] = None,
        n_pca_components: int = 3,
        lstm_num_layers: int = 3,
        lstm_hidden_dim: int = 128,
        lstm_dropout: float = 0.2,
        val_size: float = 0.2,
        test_size: float = 0.2,
        batch_size: int = 20,
        epochs: int = 100,
        rng_seed: int = None,
        base_archive_path="archive",
    ):
        r"""Initialize meta genetic algorithm.

        Args:
            fitness_function_type (MetaGAFitnessFunction): Type of fitness function of meta genetic algorithm.
            ga_generations (int): Number of generations of the genetic algorithm.
            ga_solutions_per_pop (int): Number of solutions per generation of the genetic algorithm.
            ga_percent_parents_mating (int): Percentage of parents mating for production of the offspring of the genetic algorithm [1, 100].
            ga_parent_selection_type (str): Type of parent selection of the genetic algorithm.
            ga_k_tournament (int): Number of parents participating in the tournament selection of the genetic algorithm. Only has effect when ga_parent_selection_type equals 'tournament'.
            ga_crossover_type (str): Crossover type of the genetic algorithm.
            ga_mutation_type (str): Mutation type of the genetic algorithm.
            ga_crossover_probability (float): Crossover probability of the genetic algorithm [0,1].
            ga_mutation_num_genes (int): Number of genes mutated in the solution of the genetic algorithm.
            ga_keep_elitism (int): Number of solutions that are a part of the elitism of the genetic algorithm.
            gene_spaces (dict[str | Algorithm, dict[str, Any]]): Gene spaces of the solution.
            pop_size (int): Population size of the metaheuristics being optimized.
            problem (Problem): Optimization problem used for optimization.
            max_iters (Optional[int]): Maximum number of iterations of the metaheuristic being optimized for each solution of the genetic algorithm.
            max_evals (Optional[int]): Maximum number of evaluations of the metaheuristic being optimized for each solution of the genetic algorithm.
            num_runs (Optional[int]): Number of runs performed by the metaheuristic being optimized for each solution of the genetic algorithm.
            pop_diversity_metrics (Optional[list[PopDiversityMetric]]): List of population diversity metrics calculated. Only has effect when fitness_function_type set to `*PERFORMANCE_SIMILARITY`.
            indiv_diversity_metrics (Optional[list[IndivDiversityMetric]]): List of individual diversity metrics calculated. Only has effect when fitness_function_type set to `*PERFORMANCE_SIMILARITY`.
            n_pca_components (Optional[int]): Number of PCA components to use per learning sample of the neural network. Only has effect when fitness_function_type set to `PERFORMANCE_SIMILARITY`.
            lstm_num_layers (Optional[int]): Number of layers of the LSTM neural network. Only has effect when fitness_function_type set to `PERFORMANCE_SIMILARITY`.
            lstm_hidden_dim (Optional[int]): Size of the hidden layers of the LSTM neural network. Only has effect when fitness_function_type set to `PERFORMANCE_SIMILARITY`.
            lstm_dropout (Optional[float]): Size of the dropout of the LSTM neural network [0, 1]. Only has effect when fitness_function_type set to `PERFORMANCE_SIMILARITY`.
            val_size (Optional[float]): Size of the dataset used as validation during training of the LSTM neural network. Only has effect when fitness_function_type set to `PERFORMANCE_SIMILARITY`.
            test_size (Optional[float]): Size of the dataset used as test during testing of the LSTM neural network. Only has effect when fitness_function_type set to `PERFORMANCE_SIMILARITY`.
            batch_size (Optional[int]): Size of the batch used during training of the LSTM neural network. Only has effect when fitness_function_type set to `PERFORMANCE_SIMILARITY`.
            epochs (Optional[int]): Number of epochs performed during training of the LSTM neural network. Only has effect when fitness_function_type set to `PERFORMANCE_SIMILARITY`.
            rng_seed (Optional[int]): Seed of the random generator. Provide for reproducible results. Only has effect when fitness_function_type set to `PERFORMANCE_SIMILARITY`.
            base_archive_path (Optional[str]): Base archive path of the meta genetic algorithm.
        """
        if (
            fitness_function_type
            in [
                MetaGAFitnessFunction.PERFORMANCE_SIMILARITY,
                MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY,
            ]
        ) and (pop_diversity_metrics is None or indiv_diversity_metrics is None):
            raise ValueError(
                "Diversity metrics must be defined when fitness_function_type set to `PERFORMANCE_SIMILARITY`."
            )

        if max_evals == np.inf and max_iters == np.inf:
            raise ValueError(
                "Defining a finite value for max_evals and/or max_iters is required."
            )

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
        self.n_pca_components = n_pca_components
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_dropout = lstm_dropout
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.rng_seed = rng_seed
        self.base_archive_path = base_archive_path

        self.meta_ga = None
        self.combined_gene_space = []
        self.low_ranges = []
        self.high_ranges = []
        self.random_mutation_min_val = []
        self.random_mutation_max_val = []
        self.__fitness_function = None
        self.__algorithms = []
        self.__target_algorithm = None
        self.__model_filename = "meta_ga_lstm_model.pt"
        self.__meta_dataset = "meta_dataset"
        self.__archive_path = ""
        self.__meta_ga_tmp_data_path = ""
        self.__init_parameters()

    def __init_parameters(self):
        """
        Initialize meta genetic algorithm parameters and create folder structure.
        """
        # check if all values in the provided gene spaces are correct and
        # assemble combined gene space for meta GA
        for alg_name in self.gene_spaces:
            algorithm = get_algorithm_by_name(alg_name)
            self.__algorithms.append(algorithm.Name[1])
            for setting in self.gene_spaces[alg_name]:
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
                self.combined_gene_space.append(self.gene_spaces[alg_name][setting])
                self.low_ranges.append(self.gene_spaces[alg_name][setting]["low"])
                self.high_ranges.append(self.gene_spaces[alg_name][setting]["high"])
                self.random_mutation_max_val.append(
                    abs(
                        self.gene_spaces[alg_name][setting]["high"]
                        - self.gene_spaces[alg_name][setting]["low"]
                    )
                    * 0.5
                )
                self.random_mutation_min_val.append(-self.random_mutation_max_val[-1])

        if self.fitness_function_type == MetaGAFitnessFunction.PARAMETER_TUNING:
            if len(self.__algorithms) != 1:
                raise ValueError(
                    f"Only one algorithm must be defined in the gene_spaces provided when fitness_function_type set to `PARAMETER_TUNING`."
                )
            self.__fitness_function = self.meta_ga_fitness_function_for_parameter_tuning
        elif self.fitness_function_type == MetaGAFitnessFunction.PERFORMANCE_SIMILARITY:
            if len(self.__algorithms) < 2:
                raise ValueError(
                    f"Minimum of two algorithms must be defined in gene_spaces provided when fitness_function_type set to `PERFORMANCE_SIMILARITY`."
                )
            self.__fitness_function = (
                self.meta_ga_fitness_function_for_performance_similarity
            )
        elif (
            self.fitness_function_type
            == MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY
        ):
            if len(self.__algorithms) != 1:
                raise ValueError(
                    f"Only one algorithm must be defined in gene_spaces provided when fitness_function_type set to `TARGET_PERFORMANCE_SIMILARITY`."
                )
            self.__fitness_function = (
                self.meta_ga_fitness_function_for_target_performance_similarity
            )

        # check if the provided optimization problem is correct
        if not isinstance(self.problem, Problem):
            raise TypeError(
                f"Provided problem type `{type(self.problem).__name__}` is not compatible."
            )

    def __create_folder_structure(self, prefix: str = None):
        r"""Create folder structure for the meta genetic algorithm.

        Args:
            prefix (Optional[str]): Use custom prefix for the name of the base folder in structure. Uses current datetime by default.
        """
        if prefix is None:
            prefix = str(datetime.now().strftime(f"%m-%d_%H.%M.%S"))
        self.__archive_path = os.path.join(
            self.base_archive_path,
            "_".join([prefix, *self.__algorithms, self.problem.name()]),
        )
        self.__meta_ga_tmp_data_path = os.path.join(
            self.__archive_path, "meta_ga_tmp_data"
        )
        if os.path.exists(self.__archive_path) == False:
            Path(self.__archive_path).mkdir(parents=True, exist_ok=True)

    def __get_logger(self, filename: str = "meta_ga_log_file"):
        r"""Get logger for meta genetic algorithm. Outputs to file and console.

        Args:
            filename (Optional[str]): Log file name.
        """
        level = logging.DEBUG

        logger = logging.getLogger("meta_ga_logger")
        logger.setLevel(level)
        logger.propagate = False

        file_handler = logging.FileHandler(f"{filename}.txt", "a+", "utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
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
        solution: list[float],
        gene_spaces: dict[str | Algorithm, dict[str, Any]],
        pop_size: int,
    ):
        r"""Apply meta genetic algorithm solution to an corresponding algorithms based on the gene spaces used for the meta optimization.
        Make sure the solution matches the gene space.

        Args:
            solution (list[float]): Meta genetic algorithm solution.
            gene_spaces (dict[str | Algorithm, dict[str, Any]]): Gene spaces of the solution.
            pop_size (int): Population size of the algorithms returned.

        Returns:
            list: Array of Algorithms configured based on solution and gene_space.
        """
        settings_counter = 0
        for alg_name in gene_spaces:
            settings_counter += len(gene_spaces[alg_name])
        if settings_counter != len(solution):
            raise ValueError(
                f"Solution length does not match the count of the gene space settings."
            )

        solution_iter = 0
        algorithms = []
        for alg_name in gene_spaces:
            algorithm = get_algorithm_by_name(alg_name, population_size=pop_size)

            for setting in gene_spaces[alg_name]:
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
                            algorithm.__setattr__(
                                sub_setting[1], solution[solution_iter] * sub_setting[0]
                            )
                        else:
                            if not hasattr(algorithm, sub_setting):
                                raise NameError(
                                    f"Algorithm `{alg_name}` has no attribute named `{sub_setting}`."
                                )
                            algorithm.__setattr__(sub_setting, solution[solution_iter])
                else:
                    if not hasattr(algorithm, setting):
                        raise NameError(
                            f"Algorithm `{alg_name}` has no attribute named `{setting}`."
                        )
                    algorithm.__setattr__(setting, solution[solution_iter])
                solution_iter += 1
            algorithms.append(algorithm)
        return algorithms

    @staticmethod
    def on_generation_progress(ga: pygad.GA):
        r"""Called after each genetic algorithm generation."""
        ga.logger.info(f"Generation = {ga.generations_completed}")
        ga.logger.info(
            f"->Fitness  = {ga.best_solution(pop_fitness=ga.last_generation_fitness)[1]}"
        )
        ga.logger.info(
            f"->Solution = {ga.best_solution(pop_fitness=ga.last_generation_fitness)[0]}"
        )

    def __clean_tmp_data(self):
        r"""Clean up temporary data created by the meta genetic algorithm."""
        try:
            print("Cleaning up meta GA temporary data...")
            if os.path.exists(self.__meta_ga_tmp_data_path):
                shutil.rmtree(self.__meta_ga_tmp_data_path)
        except:
            print("Cleanup failed!")

    def meta_ga_fitness_function_for_parameter_tuning(
        self, meta_ga, solution, solution_idx
    ):
        r"""Fitness function of the meta genetic algorithm.
        For tuning parameters of metaheuristic algorithms for best performance."""

        meta_dataset = os.path.join(
            self.__meta_ga_tmp_data_path, f"{solution_idx}_{self.__meta_dataset}"
        )

        algorithms = MetaGA.solution_to_algorithm_attributes(
            solution, self.gene_spaces, self.pop_size
        )

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
                    fitness_values.append(
                        SingleRunData.import_from_json(run_path).best_fitness
                    )

        avg_fitness = np.average(fitness_values)

        return 1.0 / avg_fitness + 0.0000000001

    def meta_ga_fitness_function_for_target_performance_similarity(
        self, meta_ga, solution, solution_idx
    ):
        r"""Fitness function of the meta genetic algorithm.
        For tuning parameters of metaheuristic algorithm for best similarity of diversity metrics.
        """

        meta_dataset = os.path.join(
            self.__meta_ga_tmp_data_path, f"{solution_idx}_{self.__meta_dataset}"
        )

        algorithms = MetaGA.solution_to_algorithm_attributes(
            solution, self.gene_spaces, self.pop_size
        )

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
            target_srd = SingleRunData.import_from_json(
                os.path.join(target_runs_path, target)
            )
            optimized_srd = SingleRunData.import_from_json(
                os.path.join(optimized_runs_path, optimized)
            )
            similarities.append(
                target_srd.get_diversity_metrics_similarity(optimized_srd)
            )

        return np.mean(similarities)

    def meta_ga_fitness_function_for_performance_similarity(
        self, meta_ga, solution, solution_idx
    ):
        r"""Fitness function of the meta genetic algorithm.
        For tuning parameters of metaheuristic algorithms for best similarity of diversity metrics.
        """

        model_filename = os.path.join(
            self.__meta_ga_tmp_data_path, f"{solution_idx}_{self.__model_filename}"
        )
        meta_dataset = os.path.join(
            self.__meta_ga_tmp_data_path, f"{solution_idx}_{self.__meta_dataset}"
        )

        algorithms = MetaGA.solution_to_algorithm_attributes(
            solution, self.gene_spaces, self.pop_size
        )

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
                rng_seed=self.rng_seed,
                keep_pop_data=False,
                parallel_processing=True,
            )

        train_data_loader, val_data_loader, test_data_loader, labels = get_data_loaders(
            dataset_path=meta_dataset,
            batch_size=self.batch_size,
            val_size=self.val_size,
            test_size=self.test_size,
            n_pca_components=self.n_pca_components,
            problems=[self.problem.name()],
            random_state=self.rng_seed,
        )

        # model parameters
        pop_features, indiv_features, _ = next(iter(train_data_loader))
        model = LSTMClassifier(
            input_dim=np.shape(pop_features)[2],
            aux_input_dim=np.shape(indiv_features)[1],
            num_labels=len(labels),
            hidden_dim=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        model.to(device)
        nn_train(
            model=model,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            epochs=self.epochs,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            model_filename=model_filename,
        )

        model = torch.load(model_filename, map_location=torch.device(device))
        model.to(device)
        accuracy = nn_test(model, test_data_loader, device)

        return 1.0 - accuracy + 0.0000000001

    def __before_meta_optimization(self):
        r"""Execute before meta genetic algorithm optimization."""
        if (
            self.fitness_function_type
            == MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY
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
        target_algorithm: Algorithm = None,
        get_info=False,
        prefix: str = None,
        return_best_solution: bool = False,
    ):
        r"""Run meta genetic algorithm. Saves pygad.GA instance and fitness plot image as a result of the optimization.

        Args:
            filename (Optional[str]): Name of the .pkl file of the GA object created during optimization.
            plot_filename (Optional[str]): Name of the fitness plot image file.
            target_algorithm (Optional[Algorithm]): Target algorithm for the performance similarity evaluation. Only required when fitness_function_type set to `TARGET_PERFORMANCE_SIMILARITY`.
            get_info (Optional[bool]): Generate info scheme of the meta genetic algorithm (false by default).
            prefix (Optional[str]): Use custom prefix for the name of the base folder in structure. Uses current datetime by default.
            return_best_solution (Optional[bool]): returns best solution if True.

        Returns:
            best_solution (numpy.ndarray[float] | None): Returns best solution.
        """
        self.__target_algorithm = target_algorithm
        if (
            self.fitness_function_type
            == MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY
            and self.__target_algorithm == None
        ):
            raise ValueError(
                "target_algorithm must be defined when running optimization with fitness_function_type set to `TARGET_PERFORMANCE_SIMILARITY`."
            )

        self.__create_folder_structure(prefix=prefix)
        self.__before_meta_optimization()

        if get_info:
            self.meta_ga_info(
                filename=os.path.join(self.__archive_path, "meta_ga_info"),
            )

        num_parents_mating = int(
            self.ga_solutions_per_pop * (self.ga_percent_parents_mating / 100)
        )

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
            logger=self.__get_logger(
                filename=os.path.join(self.__archive_path, "meta_ga_log_file")
            ),
        )

        self.meta_ga.run()

        self.__clean_tmp_data()

        self.meta_ga.logger.handlers.clear()
        self.meta_ga.save(os.path.join(self.__archive_path, filename))
        self.meta_ga.plot_fitness(
            save_dir=os.path.join(self.__archive_path, f"{plot_filename}.png")
        )
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
                        gene = ", ".join(
                            str(value)
                            for value in self.gene_spaces[alg_name][setting].values()
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

            if (
                self.fitness_function_type
                == MetaGAFitnessFunction.PERFORMANCE_SIMILARITY
            ):
                with c.subgraph(name="cluster_10") as cc:
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
                    for metric in self.pop_diversity_metrics:
                        pop_metrics_label += f"""<tr><td>{metric.value}</td></tr>"""
                    pop_metrics_label += "</table>>"
                    cc.node(name=f"pop_metrics", label=pop_metrics_label)

                    indiv_metrics_label = f'<<table border="0" cellborder="1" cellspacing="0"><tr><td><b>Indiv Metrics</b></td></tr>'
                    for metric in self.indiv_diversity_metrics:
                        indiv_metrics_label += f"""<tr><td>{metric.value}</td></tr>"""
                    indiv_metrics_label += "</table>>"
                    cc.node(name=f"indiv_metrics", label=indiv_metrics_label)

        if self.fitness_function_type == MetaGAFitnessFunction.PERFORMANCE_SIMILARITY:
            with gv.subgraph(name="cluster_2") as c:
                c.attr(
                    style="filled",
                    color=graph_color,
                    name="machine_learning",
                    label="Machine Learning",
                )
                c.node_attr.update(
                    style="filled",
                    color=table_border_color,
                    fillcolor=table_background_color,
                    shape="plaintext",
                    margin="0",
                )
                c.node(
                    name="ml_parameters",
                    label=f"""<
                    <table border="0" cellborder="1" cellspacing="0">
                        <tr>
                            <td colspan="2"><b>Parameters</b></td>
                        </tr>
                        <tr>
                            <td>epochs</td>
                            <td>{self.epochs}</td>
                        </tr>
                        <tr>
                            <td>batch size</td>
                            <td>{self.batch_size}</td>
                        </tr>
                        <tr>
                            <td>optimizer</td>
                            <td>Adam</td>
                        </tr>
                        <tr>
                            <td>loss</td>
                            <td>CrossEntropy</td>
                        </tr>
                    </table>>""",
                )
                c.node(
                    name="dataset_parameters",
                    label=f"""<
                    <table border="0" cellborder="1" cellspacing="0">
                        <tr>
                            <td colspan="2"><b>Dataset</b></td>
                        </tr>
                        <tr>
                            <td>size</td>
                            <td>{len(self.gene_spaces)} * {self.num_runs}</td>
                        </tr>
                        <tr>
                            <td>train</td>
                            <td>{int((1.0-self.val_size-self.test_size)*100)} %</td>
                        </tr>
                        <tr>
                            <td>val</td>
                            <td>{int(self.val_size*100)} %</td>
                        </tr>
                        <tr>
                            <td>test</td>
                            <td>{int(self.test_size*100)} %</td>
                        </tr>
                    </table>>""",
                )

                with c.subgraph(name="cluster_20") as cc:
                    cc.attr(
                        style="filled",
                        color=sub_graph_color,
                        name="architecture",
                        label="Model Architecture",
                    )
                    cc.node_attr.update(
                        style="filled",
                        color=table_border_color,
                        fillcolor=table_background_color,
                        shape="plaintext",
                        margin="0",
                    )
                    cc.node(
                        name="dense_parameters",
                        label=f"""<
                        <table border="0" cellborder="1" cellspacing="0">
                            <tr>
                                <td colspan="2"><b>Dense</b></td>
                            </tr>
                            <tr>
                                <td>input size</td>
                                <td>{self.lstm_hidden_dim + self.n_pca_components*self.pop_size}</td>
                            </tr>
                            <tr>
                                <td>output size</td>
                                <td>{len(self.gene_spaces)}</td>
                            </tr>
                        </table>>""",
                    )
                    with cc.subgraph(name="cluster_200") as ccc:
                        ccc.attr(
                            style="dashed",
                            color=table_border_color,
                            name="population",
                            label="Data loader",
                        )
                        ccc.node_attr.update(
                            style="filled",
                            color=table_border_color,
                            fillcolor=table_background_color,
                            shape="box",
                        )
                        ccc.node(
                            name="indiv_metrics_loader",
                            color=table_border_color,
                            label="Indiv Metrics",
                            margin="0.1,0,0.1,0",
                        )
                        ccc.node(
                            name="pop_metrics_loader",
                            color=table_border_color,
                            label="Pop Metrics",
                            margin="0.1,0,0.1,0",
                        )
                    cc.node(
                        name="PCA_parameters",
                        label=f"""<
                        <table border="0" cellborder="1" cellspacing="0">
                            <tr>
                                <td colspan="2"><b>PCA</b></td>
                            </tr>
                            <tr>
                                <td>components</td>
                                <td>{self.n_pca_components}</td>
                            </tr>
                            <tr>
                                <td>samples</td>
                                <td>{self.pop_size}</td>
                            </tr>
                        </table>>""",
                    )
                    cc.node(
                        name="LSTM_parameters",
                        label=f"""<
                        <table border="0" cellborder="1" cellspacing="0">
                            <tr>
                                <td colspan="2"><b>LSTM</b></td>
                            </tr>
                            <tr>
                                <td>input size</td>
                                <td>{len(self.pop_diversity_metrics)}</td>
                            </tr>
                            <tr>
                                <td>hidden size</td>
                                <td>{self.lstm_hidden_dim}</td>
                            </tr>
                            <tr>
                                <td>layers</td>
                                <td>{self.lstm_num_layers}</td>
                            </tr>
                            <tr>
                                <td>dropout</td>
                                <td>{self.lstm_dropout}</td>
                            </tr>
                        </table>>""",
                    )

                    cc.edge(
                        "LSTM_parameters",
                        "dense_parameters",
                        label=f"{self.lstm_hidden_dim}",
                    )
                    cc.edge(
                        "PCA_parameters",
                        "dense_parameters",
                        label=f"{self.n_pca_components*self.pop_size}",
                    )
                    cc.edge("pop_metrics_loader", "LSTM_parameters")
                    cc.edge("indiv_metrics_loader", "PCA_parameters")

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
            head_name=(
                "pop_metrics"
                if self.fitness_function_type
                == MetaGAFitnessFunction.PERFORMANCE_SIMILARITY
                else "optimization_parameters"
            ),
            label=f" for each {'algorithm per ' if self.fitness_function_type == MetaGAFitnessFunction.PERFORMANCE_SIMILARITY else ''} \nsolution",
            lhead="cluster_1",
        )
        gv.edge(
            tail_name="optimization_parameters",
            head_name="pop_scheme",
            ltail="cluster_1",
            lhead="cluster_3",
        )
        if self.fitness_function_type == MetaGAFitnessFunction.PERFORMANCE_SIMILARITY:
            gv.edge(
                tail_name="pop_scheme",
                head_name="dataset_parameters",
                ltail="cluster_3",
            )
        gv.edge(
            tail_name=(
                "PCA_parameters"
                if self.fitness_function_type
                == MetaGAFitnessFunction.PERFORMANCE_SIMILARITY
                else "optimization_parameters"
            ),
            head_name="combined_gene_space:gene_fitness",
            label=(
                " model accuracy\non test dataset"
                if self.fitness_function_type
                == MetaGAFitnessFunction.PERFORMANCE_SIMILARITY
                else (
                    "average fitness \nof runs"
                    if self.fitness_function_type
                    == MetaGAFitnessFunction.PARAMETER_TUNING
                    else "cosine distance \nof average feature vectors\nof target and optimized algorithm"
                )
            ),
            ltail=(
                "cluster_2"
                if self.fitness_function_type
                == MetaGAFitnessFunction.PERFORMANCE_SIMILARITY
                else "cluster_1"
            ),
        )

        gv.attr(fontsize="25")

        gv.render(format="png", cleanup=True)
