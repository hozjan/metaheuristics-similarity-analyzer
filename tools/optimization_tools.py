import multiprocessing
import threading

from niapy.algorithms import Algorithm
from niapy.problems import Problem
from niapy.task import Task
import os
import numpy as np
from numpy.random import default_rng
from pathlib import Path

from util.optimization_data import SingleRunData, PopulationData
from util.pop_diversity_metrics import PopDiversityMetric
from util.indiv_diversity_metrics import IndivDiversityMetric

__all__ = ["optimization", "optimization_worker", "optimization_runner"]


def optimization(
    algorithm: Algorithm,
    task: Task,
    single_run_data: SingleRunData,
    pop_diversity_metrics: list[PopDiversityMetric] | None = None,
    indiv_diversity_metrics: list[IndivDiversityMetric] | None = None,
    rng_seed: int | None = None,
):
    r"""An adaptation of NiaPy Algorithm run method.

    Args:
        algorithm (Algorithm): Algorithm.
        task (Task): Task with pre configured parameters.
        single_run_data (SingleRunData): Instance for archiving optimization results.
        pop_diversity_metrics (Optional[list[PopDiversityMetric]]): List of population diversity
            metrics to calculate.
        indiv_diversity_metrics (Optional[list[IndivDiversityMetric]]): List of individual diversity
            metrics to calculate.
        rng_seed (Optional[int]): Seed for the rng, provide for reproducible results.

        Returns:
            Tuple[numpy.ndarray | None, float | None]:
                1. Best individuals components found in optimization process.
                2. Best fitness value found in optimization process.

        Raises:
            BaseException: Algorithm exception.
    """
    try:
        algorithm.callbacks.before_run()
        if rng_seed is not None:
            algorithm.rng = default_rng(seed=rng_seed)
        else:
            algorithm.rng = default_rng()

        pop, fpop, params = algorithm.init_population(task)

        if rng_seed is not None:
            algorithm.rng = default_rng()

        xb, fxb = algorithm.get_best(pop, fpop)

        order_idx = np.array([*range(algorithm.population_size)])
        sorted_idx = order_idx.copy()

        while not task.stopping_condition():
            # save population data
            pop_data = PopulationData(
                population=np.array(pop[sorted_idx]),
                population_fitness=np.array(fpop[sorted_idx]),
                best_solution=np.array(xb),
                best_fitness=fxb * task.optimization_type.value,
            )
            if pop_diversity_metrics is not None:
                pop_data.calculate_metrics(
                    pop_diversity_metrics,
                    task.problem,
                )
            single_run_data.add_population(pop_data)
            algorithm.callbacks.before_iteration(pop, fpop, xb, fxb, **params)
            pop, fpop, xb, fxb, params = algorithm.run_iteration(task, pop, fpop, xb, fxb, **params)

            if "sorted_idx" in params:
                order_idx = order_idx[params.get("sorted_idx")]
                sorted_idx = np.argsort(order_idx)

            algorithm.callbacks.after_iteration(pop, fpop, xb, fxb, **params)
            task.next_iter()
        algorithm.callbacks.after_run()
        if indiv_diversity_metrics is not None:
            single_run_data.calculate_indiv_diversity_metrics(indiv_diversity_metrics)
        single_run_data.evals = task.evals
        return xb, fxb * task.optimization_type.value
    except BaseException as e:
        if (
            threading.current_thread() is threading.main_thread()
            and multiprocessing.current_process().name == "MainProcess"
        ):
            raise e
        algorithm.exception = e
        return None, None


def optimization_worker(
    problem: Problem,
    algorithm: Algorithm,
    pop_diversity_metrics: list[PopDiversityMetric] | None = None,
    indiv_diversity_metrics: list[IndivDiversityMetric] | None = None,
    max_iters: int | float = np.inf,
    max_evals: int | float = np.inf,
    dataset_path: str | None = None,
    rng_seed: int | None = None,
    run_index: int | None = None,
    keep_pop_data: bool = True,
    keep_diversity_metrics: bool = True,
):
    r"""Single optimization run execution.

    Args:
        algorithm (Algorithm): Algorithm.
        problem (Problem): Optimization problem.
        pop_diversity_metrics (Optional[list[PopDiversityMetric]]): List of population diversity
            metrics to calculate.
        indiv_diversity_metrics (Optional[list[IndivDiversityMetric]]): List of individual diversity
            metrics to calculate.
        max_iters (Optional[int | float]): Individual optimization run stopping condition.
        max_evals (Optional[int | float]): Individual optimization run stopping condition.
        dataset_path (Optional[str]): Path to the dataset to be created.
        rng_seed (Optional[int]): Seed for the random generator, provide for reproducible results.
        run_index (Optional[int]): Run index, used for file name indexing. Has no effect if dataset_path is None.
        keep_pop_data (Optional[bool]): If false clear population solutions and fitness values in order to save space
            on data export. Does not clear diversity metrics. Has no effect if dataset_path is None.
        keep_diversity_metrics (Optional[bool]): If false clear diversity metrics in order to further save space on
            data export. Has no effect if keep_pop_data is true (true by default).

        Raises:
            IndexError: `run_index` was not provided when required.
    """
    task = Task(problem, max_iters=max_iters, max_evals=max_evals)

    single_run_data = SingleRunData(
        algorithm_name=algorithm.Name,
        algorithm_parameters=algorithm.get_parameters(),
        problem_name=problem.name(),
        max_iters=max_iters,
        max_evals=max_evals,
        rng_seed=rng_seed,
    )

    optimization(
        algorithm=algorithm,
        task=task,
        single_run_data=single_run_data,
        pop_diversity_metrics=pop_diversity_metrics,
        indiv_diversity_metrics=indiv_diversity_metrics,
        rng_seed=rng_seed,
    )

    if dataset_path is None:
        return single_run_data

    if run_index is None:
        raise IndexError("Run index must be provided in order to generate a valid dataset.")
    # check if folder structure exists, if not create it
    path = os.path.join(dataset_path, algorithm.Name[1], problem.name())
    if os.path.exists(path) is False:
        Path(path).mkdir(parents=True, exist_ok=True)

    single_run_data.export_to_json(
        os.path.join(path, f"run_{run_index:05d}.json"),
        keep_pop_data=keep_pop_data,
        keep_diversity_metrics=keep_diversity_metrics,
    )

    return single_run_data


def optimization_runner(
    algorithm: Algorithm,
    problem: Problem,
    runs: int,
    dataset_path: str,
    pop_diversity_metrics: list[PopDiversityMetric] | None = None,
    indiv_diversity_metrics: list[IndivDiversityMetric] | None = None,
    max_iters: int | float = np.inf,
    max_evals: int | float = np.inf,
    rng_seed: int | None = None,
    run_index_seed: bool = False,
    keep_pop_data: bool = True,
    keep_diversity_metrics: bool = True,
    parallel_processing: bool = False,
):
    r"""Optimization work splitter.

    Args:
        algorithm (Algorithm): Algorithm.
        problem (Problem): Optimization problem.
        runs (int): Number of runs to execute.
        dataset_path (str): Path to the dataset to be created.
        pop_diversity_metrics (Optional[list[PopDiversityMetric]]): List of population diversity
            metrics to calculate.
        indiv_diversity_metrics (Optional[list[IndivDiversityMetric]]): List of individual diversity
            metrics to calculate.
        max_iters (Optional[int]): Individual optimization run stopping condition.
        max_evals (Optional[int]): Individual optimization run stopping condition.
        rng_seed (Optional[int]): Seed for the rng, provide for reproducible results.
            Has no effect if run_index_seed is True.
        run_index_seed (Optional[bool]): Use run index as rng_seed for increased but controlled randomization.
        keep_pop_data (Optional[bool]): If false clear population solutions and fitness values in order to save space
            on data export. Does not clear diversity metrics.
        keep_diversity_metrics (Optional[bool]): If false clear diversity metrics in order to further save space on
            data export. Has no effect if keep_pop_data is true (true by default).
        parallel_processing (Optional[bool]): Execute optimization runs in parallel over multiple processes.
    """
    if parallel_processing:
        pool = []
        for r_idx in range(runs):
            if run_index_seed:
                rng_seed = r_idx
            p = multiprocessing.Process(
                target=optimization_worker,
                args=(
                    problem,
                    algorithm,
                    pop_diversity_metrics,
                    indiv_diversity_metrics,
                    max_iters,
                    max_evals,
                    dataset_path,
                    rng_seed,
                    r_idx,
                    keep_pop_data,
                    keep_diversity_metrics,
                ),
            )
            p.start()
            pool.append(p)

        for p in pool:
            p.join()
    else:
        for r_idx in range(runs):
            if run_index_seed:
                rng_seed = r_idx
            optimization_worker(
                problem=problem,
                algorithm=algorithm,
                pop_diversity_metrics=pop_diversity_metrics,
                indiv_diversity_metrics=indiv_diversity_metrics,
                max_iters=max_iters,
                max_evals=max_evals,
                dataset_path=dataset_path,
                rng_seed=rng_seed,
                run_index=r_idx,
                keep_pop_data=keep_pop_data,
                keep_diversity_metrics=keep_diversity_metrics,
            )
