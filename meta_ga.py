from typing import Any
from niapy.util.factory import (
    _algorithm_options,
    _problem_options,
    get_algorithm,
    get_problem,
)
from niapy.problems import Problem
from datetime import datetime
import torch
from torch import nn
import numpy as np
import pygad
from tools.optimization_tools import optimization_runner, optimization_worker
from tools.ml_tools import get_data_loaders, nn_test, nn_train, LSTM
import shutil
import logging
import os

from util.constants import (
    RNG_SEED,
    BATCH_SIZE,
    EPOCHS,
    POP_SIZE,
    MAX_ITERS,
    NUM_RUNS,
    META_GA_GENERATIONS,
    OPTIMIZATION_PROBLEM,
    GENE_SPACES,
    META_GA_SOLUTIONS_PER_POP,
    POP_DIVERSITY_METRICS,
    INDIV_DIVERSITY_METRICS,
)

META_GA_TMP_DATA = "meta_ga_tmp_data"
MODEL_FILE_NAME = os.path.join(META_GA_TMP_DATA, "meta_ga_lstm_model.pt")
META_DATASET_PATH = os.path.join(META_GA_TMP_DATA, "meta_dataset")

__all__ = [
    "meta_ga_fitness_function",
    "clean_tmp_data",
    "solution_to_algorithm_attributes",
    "run_meta_ga",
]


def meta_ga_fitness_function(meta_ga, solution, solution_idx):
    r"""Fitness function of the meta genetic algorithm."""

    algorithms = solution_to_algorithm_attributes(solution, GENE_SPACES)

    if isinstance(OPTIMIZATION_PROBLEM, Problem):
        problem = OPTIMIZATION_PROBLEM
    else:
        problem = get_problem(OPTIMIZATION_PROBLEM)

    # gather optimization data
    for algorithm in algorithms:
        optimization_runner(
            algorithm,
            problem,
            NUM_RUNS,
            META_DATASET_PATH,
            POP_DIVERSITY_METRICS,
            INDIV_DIVERSITY_METRICS,
            max_iters=MAX_ITERS,
            rng_seed=RNG_SEED,
            parallel_processing=True,
        )

    train_data_loader, val_data_loader, test_data_loader, labels = get_data_loaders(
        META_DATASET_PATH,
        BATCH_SIZE,
        problems=[
            (
                OPTIMIZATION_PROBLEM.name()
                if isinstance(OPTIMIZATION_PROBLEM, Problem)
                else OPTIMIZATION_PROBLEM
            )
        ],
        random_state=RNG_SEED,
    )

    # model parameters
    pop_features, indiv_features, _ = next(iter(train_data_loader))
    model = LSTM(np.shape(pop_features)[2], np.shape(indiv_features)[1], len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    nn_train(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        epochs=EPOCHS,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        patience=np.inf,
        model_file_name=MODEL_FILE_NAME,
    )

    model = torch.load(MODEL_FILE_NAME, map_location=torch.device(device))
    model.to(device)
    accuracy = nn_test(model, test_data_loader, device)

    return 1.0 - accuracy


def ed_meta_ga_fitness_function(meta_ga, solution, solution_idx):
    r"""Fitness function of the meta genetic algorithm based on euclidean distance."""

    algorithms = solution_to_algorithm_attributes(solution, GENE_SPACES)

    if isinstance(OPTIMIZATION_PROBLEM, Problem):
        problem = OPTIMIZATION_PROBLEM
    else:
        problem = get_problem(OPTIMIZATION_PROBLEM)

    # gather optimization data
    data = []
    for algorithm in algorithms:
        algorithm_data = []
        for _ in range(NUM_RUNS):
            single_run_data = optimization_worker(
                problem=problem,
                algorithm=algorithm,
                pop_diversity_metrics=POP_DIVERSITY_METRICS,
                indiv_diversity_metrics=INDIV_DIVERSITY_METRICS,
                max_iters=MAX_ITERS,
                rng_seed=RNG_SEED,
            )
            algorithm_data.append(single_run_data)
        data.append(algorithm_data)

    euclidean_distance = 0

    for first in data[0]:
        for second in data[1]:
            euclidean_distance += first.diversity_metrics_euclidean_distance(
                second, include_fitness_convergence=True
            )

    euclidean_distance /= pow(NUM_RUNS, len(algorithms))

    return 1.0 / (euclidean_distance + 0.0000000001)


def clean_tmp_data():
    r"""Clean up temporary data created by the meta genetic algorithm."""
    try:
        print("Cleaning up meta GA temporary data...")
        if os.path.exists(META_GA_TMP_DATA):
            shutil.rmtree(META_GA_TMP_DATA)
    except:
        print("Cleanup failed!")


def on_generation_progress(ga: pygad.GA):
    r"""Called after each genetic algorithm generation."""
    ga.logger.info(f"Generation = {ga.generations_completed}")
    ga.logger.info(
        f"->Fitness  = {ga.best_solution(pop_fitness=ga.last_generation_fitness)[1]}"
    )
    ga.logger.info(
        f"->Solution = {ga.best_solution(pop_fitness=ga.last_generation_fitness)[0]}"
    )


def solution_to_algorithm_attributes(
    solution: list[float], gene_spaces: dict[str, dict[str, Any]]
):
    r"""Apply meta genetic algorithm solution to an corresponding algorithm based on the gene spaces used for the meta optimization.
    Make sure the solution matches the gene space.

    Args:
        solution (list[float]): Meta genetic algorithm solution.
        gene_spaces (dict[str, dict[str, Any]]): Gene spaces of the solution.

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
        algorithm = get_algorithm(alg_name, population_size=POP_SIZE)
        for setting in gene_spaces[alg_name]:
            algorithm.__setattr__(setting, solution[solution_iter])
            solution_iter += 1
        algorithms.append(algorithm)
    return algorithms


def run_meta_ga(filename="meta_ga_obj", plot_filename="meta_ga_fitness_plot"):
    r"""Run meta genetic algorithm. Saves pygad.GA instance and fitness plot image as a result of optimization.

    Args:
        filename (Optional[str]): Name of the .pkl file of the GA object created during optimization.
        plot_filename (Optional[str]): Name of the fitness plot image file.
    """

    combined_gene_space = []
    low_ranges = []
    high_ranges = []
    # check if all values in the provided gene spaces are correct and
    # assemble combined gene space for meta GA
    for alg_name in GENE_SPACES:
        if alg_name not in _algorithm_options():
            raise KeyError(
                f"Could not find algorithm by name `{alg_name}` in the niapy library."
            )
        algorithm = get_algorithm(alg_name)
        for setting in GENE_SPACES[alg_name]:
            if not hasattr(algorithm, setting):
                raise NameError(
                    f"Algorithm `{alg_name}` has no attribute named `{setting}`."
                )
            combined_gene_space.append(GENE_SPACES[alg_name][setting])
            low_ranges.append(GENE_SPACES[alg_name][setting]["low"])
            high_ranges.append(GENE_SPACES[alg_name][setting]["high"])

    # check if the provided optimization problem is correct
    if (
        not isinstance(OPTIMIZATION_PROBLEM, Problem)
        and OPTIMIZATION_PROBLEM.lower() not in _problem_options()
    ):
        raise KeyError(
            f"Could not find optimization problem by name `{OPTIMIZATION_PROBLEM}` in the niapy library."
        )

    clean_tmp_data()

    level = logging.DEBUG
    name = "./meta_ga_logfile.txt"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(name, "a+", "utf-8")
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

    meta_ga = pygad.GA(
        num_generations=META_GA_GENERATIONS,
        num_parents_mating=int(META_GA_SOLUTIONS_PER_POP * 0.4),
        fitness_func=ed_meta_ga_fitness_function,
        sol_per_pop=META_GA_SOLUTIONS_PER_POP,
        num_genes=len(combined_gene_space),
        parent_selection_type="rws",
        init_range_low=low_ranges,
        init_range_high=high_ranges,
        crossover_type="two_points",
        mutation_type="random",
        mutation_percent_genes=30,
        gene_space=combined_gene_space,
        on_generation=on_generation_progress,
        save_best_solutions=True,
        stop_criteria="saturate_10",
        parallel_processing=["process", 100],
        logger=logger,
    )

    meta_ga.run()

    clean_tmp_data()

    meta_ga.save(filename)
    meta_ga.plot_fitness(save_dir=f"{plot_filename}.png")
    best_solutions = meta_ga.best_solutions
    print(f"Best solution: {best_solutions[-1]}")


if __name__ == "__main__":
    start = datetime.now()
    run_meta_ga()
    end = datetime.now()
    elapsed = end - start
    print(f"Time elapsed: {elapsed}")
