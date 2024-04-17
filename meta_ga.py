from niapy.util.factory import _algorithm_options, _problem_options, get_algorithm, get_problem
from niapy.problems.schwefel import Schwefel
import torch
import tqdm
from torch import nn
import numpy as np
import pygad
from tools.optimization_tools import optimization_runner
from tools.ml_tools import get_data_loaders, nn_test, nn_train, LSTM
import shutil
import os

from util.constants import (
    RNG_SEED,
    BATCH_SIZE,
    EPOCHS,
    POP_SIZE,
    MAX_ITER,
    NUM_RUNS,
    META_GA_GENERATIONS,
    OPTIMIZATION_PROBLEM,
    GENE_SPACES,
    META_GA_SOLUTIONS_PER_POP
)

MODEL_FILE_NAME = "meta_ga_lstm_model.pt"
META_DATASET_PATH = "./meta_dataset"

def meta_ga_fitness_function(meta_ga, solution, solution_idx):
    solution_iter = 0
    algorithms = []
    for alg_name in GENE_SPACES:
        algorithm = get_algorithm(alg_name, population_size=POP_SIZE)
        for setting in GENE_SPACES[alg_name]:
            algorithm.__setattr__(setting, solution[solution_iter])
            solution_iter += 1
        algorithms.append(algorithm)

    problem = get_problem(OPTIMIZATION_PROBLEM)
    # gather optimization data
    for algorithm in algorithms:
        optimization_runner(
            algorithm, problem, MAX_ITER, NUM_RUNS, META_DATASET_PATH, rng_seed=RNG_SEED, parallel_processing=True
        )

    train_data_loader, val_data_loader, test_data_loader, labels = get_data_loaders(
        META_DATASET_PATH, BATCH_SIZE, problems=[OPTIMIZATION_PROBLEM], random_state=RNG_SEED
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
        model_file_name=MODEL_FILE_NAME,
    )

    model = torch.load(MODEL_FILE_NAME, map_location=torch.device(device))
    model.to(device)
    accuracy = nn_test(model, test_data_loader, device)

    return 1.0 - accuracy

def on_generation_progress(ga):
        progress_bar.update(1)

if __name__ == "__main__":
    combined_gene_space = []
    low_ranges = []
    high_ranges = []
    # check if all values in the provided gene spaces are correct and
    # assemble combined gene space for meta GA
    for alg_name in GENE_SPACES:
        if alg_name not in _algorithm_options():
            raise KeyError(f"Could not find algorithm by name `{alg_name}` in the niapy library.")
        algorithm = get_algorithm(alg_name)
        for setting in GENE_SPACES[alg_name]:
            if not hasattr(algorithm, setting):
                raise NameError(f"Algorithm `{alg_name}` has no attribute named `{setting}`.")
            combined_gene_space.append(GENE_SPACES[alg_name][setting])
            low_ranges.append(GENE_SPACES[alg_name][setting]["low"])
            high_ranges.append(GENE_SPACES[alg_name][setting]["high"])

    # check if the provided optimization problem is correct
    if OPTIMIZATION_PROBLEM.lower() not in _problem_options():
        raise KeyError(f"Could not find optimization problem by name `{OPTIMIZATION_PROBLEM}` in the niapy library.")

    with tqdm.tqdm(total=META_GA_GENERATIONS) as progress_bar:
        meta_ga = pygad.GA(
            num_generations=META_GA_GENERATIONS,
            num_parents_mating=2,
            fitness_func=meta_ga_fitness_function,
            sol_per_pop=META_GA_SOLUTIONS_PER_POP,
            num_genes=len(combined_gene_space),
            init_range_low=low_ranges,
            init_range_high=high_ranges,
            parent_selection_type="sss",
            keep_parents=1,
            crossover_type="two_points",
            mutation_type="random",
            mutation_percent_genes=10,
            gene_space=combined_gene_space,
            on_generation=on_generation_progress,
            save_best_solutions=True,
        )

        meta_ga.run()

    try:
        print("Removing temporary data...")
        shutil.rmtree(META_DATASET_PATH)
        os.remove(MODEL_FILE_NAME)
    except:
        print("Removing temporary data failed!")

    meta_ga.save("meta_ga_instance")
    meta_ga.plot_fitness(save_dir="meta_ga_fitness_plot.png")
    best_solutions = meta_ga.best_solutions
    print(f"Best solution: {best_solutions[-1]}")
