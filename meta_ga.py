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
from util.optimization_data import SingleRunData
import shutil
import logging
import graphviz
import os

from util.constants import (
    RNG_SEED,
    BATCH_SIZE,
    EPOCHS,
    POP_SIZE,
    MAX_ITERS,
    NUM_RUNS,
    META_GA_GENERATIONS,
    META_GA_PERCENT_PARENTS_MATING,
    META_GA_SOLUTIONS_PER_POP,
    META_GA_PARENT_SELECTION_TYPE,
    META_GA_K_TOURNAMENT,
    META_GA_CROSSOVER_TYPE,
    META_GA_MUTATION_TYPE,
    META_GA_CROSSOVER_PROBABILITY,
    META_GA_MUTATION_NUM_GENES,
    META_GA_KEEP_ELITISM,
    OPTIMIZATION_PROBLEM,
    GENE_SPACES,
    POP_DIVERSITY_METRICS,
    INDIV_DIVERSITY_METRICS,
    N_PCA_COMPONENTS,
    LSTM_NUM_LAYERS,
    LSTM_HIDDEN_DIM,
    LSTM_DROPOUT,
    VAL_SIZE,
    TEST_SIZE,
)

META_GA_TMP_DATA = "meta_ga_tmp_data"
MODEL_FILE_NAME = "meta_ga_lstm_model.pt"
META_DATASET = "meta_dataset"

__all__ = [
    "meta_ga_info",
    "meta_ga_fitness_function",
    "clean_tmp_data",
    "solution_to_algorithm_attributes",
    "run_meta_ga",
]


def meta_ga_info(
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
                    <td>{META_GA_GENERATIONS}</td>
                </tr>
                <tr>
                    <td>pop size</td>
                    <td>{META_GA_SOLUTIONS_PER_POP}</td>
                </tr>
                <tr>
                    <td>parent selection</td>
                    <td>{META_GA_PARENT_SELECTION_TYPE}</td>
                </tr>
                """
        
        if META_GA_PARENT_SELECTION_TYPE == "tournament":
            meta_ga_parameters_label += f"""
                <tr>
                    <td>K tournament</td>
                    <td>{META_GA_K_TOURNAMENT}</td>
                </tr>"""
        meta_ga_parameters_label += f"""
                <tr>
                    <td>parents</td>
                    <td>{META_GA_PERCENT_PARENTS_MATING} %</td>
                </tr>
                <tr>
                    <td>crossover type</td>
                    <td>{META_GA_CROSSOVER_TYPE}</td>
                </tr>
                <tr>
                    <td>mutation type</td>
                    <td>{META_GA_MUTATION_TYPE}</td>
                </tr>
                <tr>
                    <td>crossover prob.</td>
                    <td>{META_GA_CROSSOVER_PROBABILITY}</td>
                </tr>
                <tr>
                    <td>mutate num genes</td>
                    <td>{META_GA_MUTATION_NUM_GENES}</td>
                </tr>
                <tr>
                    <td>keep elitism</td>
                    <td>{META_GA_KEEP_ELITISM}</td>
                </tr>
                <tr>
                    <td>rng seed</td>
                    <td>{RNG_SEED}</td>
                </tr>
            </table>>"""
        c.node(
            name="meta_ga_parameters",
            label=meta_ga_parameters_label
        )

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
            for alg_idx, alg_name in enumerate(list(GENE_SPACES.keys())):
                node_label = f"""<<table border="0" cellborder="1" cellspacing="0">
                    <tr>
                        <td colspan="2"><b>{alg_name}</b></td>
                    </tr>
                    <tr>
                        <td>pop size</td>
                        <td>{POP_SIZE}</td>
                    </tr>"""
                for setting in GENE_SPACES[alg_name]:
                    gene = ", ".join(
                        str(value) for value in GENE_SPACES[alg_name][setting].values()
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

            for alg_idx in range(len(list(GENE_SPACES.keys()))):
                cc.edge(f"gene_space_{alg_idx}", "combined_gene_space")

    with gv.subgraph(name="cluster_1") as c:
        c.attr(
            style="filled", color=graph_color, name="optimization", label="Optimization"
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
                    <td>{MAX_ITERS}</td>
                </tr>
                <tr>
                    <td>num runs</td>
                    <td>{NUM_RUNS}</td>
                </tr>
                <tr>
                    <td>problem</td>
                    <td>{OPTIMIZATION_PROBLEM.name()}</td>
                </tr>
                <tr>
                    <td>dimension</td>
                    <td>{OPTIMIZATION_PROBLEM.dimension}</td>
                </tr>
            </table>>""",
        )

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
            for metric in POP_DIVERSITY_METRICS:
                pop_metrics_label += f"""<tr><td>{metric.value}</td></tr>"""
            pop_metrics_label += "</table>>"
            cc.node(name=f"pop_metrics", label=pop_metrics_label)

            indiv_metrics_label = f'<<table border="0" cellborder="1" cellspacing="0"><tr><td><b>Indiv Metrics</b></td></tr>'
            for metric in INDIV_DIVERSITY_METRICS:
                indiv_metrics_label += f"""<tr><td>{metric.value}</td></tr>"""
            indiv_metrics_label += "</table>>"
            cc.node(name=f"indiv_metrics", label=indiv_metrics_label)

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
                    <td>{EPOCHS}</td>
                </tr>
                <tr>
                    <td>batch size</td>
                    <td>{BATCH_SIZE}</td>
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
                    <td>{len(list(GENE_SPACES.keys()))} * {NUM_RUNS}</td>
                </tr>
                <tr>
                    <td>train</td>
                    <td>{int((1.0-VAL_SIZE-TEST_SIZE)*100)} %</td>
                </tr>
                <tr>
                    <td>val</td>
                    <td>{int(VAL_SIZE*100)} %</td>
                </tr>
                <tr>
                    <td>test</td>
                    <td>{int(TEST_SIZE*100)} %</td>
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
                        <td>{LSTM_HIDDEN_DIM + N_PCA_COMPONENTS*POP_SIZE}</td>
                    </tr>
                    <tr>
                        <td>output size</td>
                        <td>{len(list(GENE_SPACES.keys()))}</td>
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
                        <td>{N_PCA_COMPONENTS}</td>
                    </tr>
                    <tr>
                        <td>samples</td>
                        <td>{POP_SIZE}</td>
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
                        <td>{len(POP_DIVERSITY_METRICS)}</td>
                    </tr>
                    <tr>
                        <td>hidden size</td>
                        <td>{LSTM_HIDDEN_DIM}</td>
                    </tr>
                    <tr>
                        <td>layers</td>
                        <td>{LSTM_NUM_LAYERS}</td>
                    </tr>
                    <tr>
                        <td>dropout</td>
                        <td>{LSTM_DROPOUT}</td>
                    </tr>
                </table>>""",
            )

            cc.edge("LSTM_parameters", "dense_parameters", label=f"{LSTM_HIDDEN_DIM}")
            cc.edge(
                "PCA_parameters",
                "dense_parameters",
                label=f"{N_PCA_COMPONENTS*POP_SIZE}",
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
                                <td><i><b>X</b><sub>i=1, t={MAX_ITERS}</sub></i></td>
                            </tr>
                            <tr>
                                <td>...</td>
                                <td>...</td>
                                <td>...</td>
                            </tr>
                            <tr>
                                <td><i><b>X</b><sub>i={POP_SIZE}, t=1</sub></i></td>
                                <td>...</td>
                                <td><i><b>X</b><sub>i={POP_SIZE}, t={MAX_ITERS}</sub></i></td>
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
                                <td><i><b>IM</b><sub>{POP_SIZE}</sub></i></td>
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
                                <td><i><b>PM</b><sub>{MAX_ITERS}</sub></i></td>
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
        label=" for each algorithm\nper solution",
        lhead="cluster_1",
    )
    gv.edge(
        tail_name="optimization_parameters",
        head_name="pop_scheme",
        ltail="cluster_1",
        lhead="cluster_3",
    )
    gv.edge(tail_name="pop_scheme", head_name="dataset_parameters", ltail="cluster_3")
    gv.edge(
        tail_name="PCA_parameters",
        head_name="combined_gene_space:gene_fitness",
        label=" model accuracy\non test dataset",
        ltail="cluster_2",
    )

    gv.attr(fontsize="25")

    gv.render(format="png", cleanup=True)


def meta_ga_fitness_function_for_parameter_tuning(meta_ga, solution, solution_idx):
    r"""Fitness function of the meta genetic algorithm.
    For tuning parameters of metaheuristic algorithms for best performance."""

    _META_DATASET = os.path.join(META_GA_TMP_DATA, f"{solution_idx}_{META_DATASET}")

    algorithms = solution_to_algorithm_attributes(solution, GENE_SPACES)

    if isinstance(OPTIMIZATION_PROBLEM, Problem):
        problem = OPTIMIZATION_PROBLEM
    else:
        problem = get_problem(OPTIMIZATION_PROBLEM)

    # gather optimization data
    for algorithm in algorithms:
        optimization_runner(
            algorithm=algorithm,
            problem=problem,
            runs=NUM_RUNS,
            dataset_path=_META_DATASET,
            max_iters=MAX_ITERS,
            run_index_seed=True,
            keep_pop_data=False,
            keep_diversity_metrics=False,
            parallel_processing=True,
        )

    fitness_values = []
    for algorithm in os.listdir(_META_DATASET):
        for problem in os.listdir(os.path.join(_META_DATASET, algorithm)):
            runs = os.listdir(os.path.join(_META_DATASET, algorithm, problem))
            for run in runs:
                run_path = os.path.join(_META_DATASET, algorithm, problem, run)
                fitness_values.append(SingleRunData.import_from_json(run_path).best_fitness)

    avg_fitness = np.average(fitness_values)

    return 1.0 / avg_fitness + 0.0000000001

def meta_ga_fitness_function(meta_ga, solution, solution_idx):
    r"""Fitness function of the meta genetic algorithm."""

    _MODEL_FILE_NAME = os.path.join(
        META_GA_TMP_DATA, f"{solution_idx}_{MODEL_FILE_NAME}"
    )
    _META_DATASET = os.path.join(META_GA_TMP_DATA, f"{solution_idx}_{META_DATASET}")

    algorithms = solution_to_algorithm_attributes(solution, GENE_SPACES)

    if isinstance(OPTIMIZATION_PROBLEM, Problem):
        problem = OPTIMIZATION_PROBLEM
    else:
        problem = get_problem(OPTIMIZATION_PROBLEM)

    # gather optimization data
    for algorithm in algorithms:
        optimization_runner(
            algorithm=algorithm,
            problem=problem,
            runs=NUM_RUNS,
            dataset_path=_META_DATASET,
            pop_diversity_metrics=POP_DIVERSITY_METRICS,
            indiv_diversity_metrics=INDIV_DIVERSITY_METRICS,
            max_iters=MAX_ITERS,
            rng_seed=RNG_SEED,
            keep_pop_data=False,
            parallel_processing=True,
        )

    train_data_loader, val_data_loader, test_data_loader, labels = get_data_loaders(
        dataset_path=_META_DATASET,
        batch_size=BATCH_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        n_pca_components=N_PCA_COMPONENTS,
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
    model = LSTM(
        input_dim=np.shape(pop_features)[2],
        aux_input_dim=np.shape(indiv_features)[1],
        num_labels=len(labels),
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    )
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
        model_file_name=_MODEL_FILE_NAME,
    )

    model = torch.load(_MODEL_FILE_NAME, map_location=torch.device(device))
    model.to(device)
    accuracy = nn_test(model, test_data_loader, device)

    return 1.0 - accuracy + 0.0000000001


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


def get_logger(filename: str = "meta_ga_log_file"):
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
    random_mutation_min_val = []
    random_mutation_max_val = []
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
            random_mutation_max_val.append(abs(GENE_SPACES[alg_name][setting]["high"] - GENE_SPACES[alg_name][setting]["low"]) * 0.5)
            random_mutation_min_val.append(-random_mutation_max_val[-1])

    # check if the provided optimization problem is correct
    if (
        not isinstance(OPTIMIZATION_PROBLEM, Problem)
        and OPTIMIZATION_PROBLEM.lower() not in _problem_options()
    ):
        raise KeyError(
            f"Could not find optimization problem by name `{OPTIMIZATION_PROBLEM}` in the niapy library."
        )

    clean_tmp_data()

    num_parents_mating = int(
        META_GA_SOLUTIONS_PER_POP * (META_GA_PERCENT_PARENTS_MATING / 100)
    )

    meta_ga = pygad.GA(
        num_generations=META_GA_GENERATIONS,
        num_parents_mating=num_parents_mating,
        keep_elitism=META_GA_KEEP_ELITISM,
        allow_duplicate_genes=False,
        fitness_func=meta_ga_fitness_function_for_parameter_tuning,
        sol_per_pop=META_GA_SOLUTIONS_PER_POP,
        num_genes=len(combined_gene_space),
        parent_selection_type=META_GA_PARENT_SELECTION_TYPE,
        K_tournament=META_GA_K_TOURNAMENT,
        init_range_low=low_ranges,
        init_range_high=high_ranges,
        crossover_type=META_GA_CROSSOVER_TYPE,
        crossover_probability=META_GA_CROSSOVER_PROBABILITY,
        mutation_type=META_GA_MUTATION_TYPE,
        mutation_num_genes=META_GA_MUTATION_NUM_GENES,
        random_mutation_min_val=random_mutation_min_val,
        random_mutation_max_val=random_mutation_max_val,
        gene_space=combined_gene_space,
        on_generation=on_generation_progress,
        save_best_solutions=True,
        save_solutions=True,
        stop_criteria="saturate_10",
        parallel_processing=["process", META_GA_SOLUTIONS_PER_POP],
        logger=get_logger(),
    )

    meta_ga.run()

    clean_tmp_data()

    meta_ga.save(filename)
    meta_ga.plot_fitness(save_dir=f"{plot_filename}.png")
    best_solutions = meta_ga.best_solutions
    print(f"Best solution: {best_solutions[-1]}")


if __name__ == "__main__":
    meta_ga_info()
    start = datetime.now()
    run_meta_ga()
    end = datetime.now()
    elapsed = end - start
    print(f"Time elapsed: {elapsed}")
