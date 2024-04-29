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
    pop_diversity_metrics: list[PopDiversityMetric],
    indiv_diversity_metrics: list[IndivDiversityMetric],
    rng_seed: int = None,
):
    r"""An adaptation of NiaPy Algorithm run method.

    Args:
        algorithm (Algorithm): Algorithm.
        task (Task): Task with pre configured parameters.
        single_run_data (SingleRunData): Instance for archiving optimization results
        pop_diversity_metrics (list[PopDiversityMetric]): List of population diversity metrics to calculate.
        indiv_diversity_metrics (list[IndivDiversityMetric]): List of individual diversity metrics to calculate.
        rng_seed (Optional[int]): Seed for the rng, provide for reproducible results.
    """
    try:
        algorithm.callbacks.before_run()
        if rng_seed is not None:
            algorithm.rng = default_rng(seed=rng_seed)

        pop, fpop, params = algorithm.init_population(task)

        if rng_seed is not None:
            algorithm.rng = default_rng()

        xb, fxb = algorithm.get_best(pop, fpop)
        while not task.stopping_condition():
            # save population data
            pop_data = PopulationData(
                population=np.array(pop), population_fitness=np.array(fpop)
            )
            pop_data.calculate_metrics(
                pop_diversity_metrics,
                task.problem,
            )
            single_run_data.add_population(pop_data)
            algorithm.callbacks.before_iteration(pop, fpop, xb, fxb, **params)
            pop, fpop, xb, fxb, params = algorithm.run_iteration(
                task, pop, fpop, xb, fxb, **params
            )

            algorithm.callbacks.after_iteration(pop, fpop, xb, fxb, **params)
            task.next_iter()
        algorithm.callbacks.after_run()
        single_run_data.calculate_indiv_diversity_metrics(indiv_diversity_metrics)
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
    run_index: int,
    dataset_path: str,
    pop_diversity_metrics: list[PopDiversityMetric],
    indiv_diversity_metrics: list[IndivDiversityMetric],
    max_iters: int = np.inf,
    max_evals: int = np.inf,
    rng_seed: int = None,
):
    r"""Single optimization run execution.

    Args:
        algorithm (Algorithm): Algorithm.
        problem (Problem): Optimization problem.
        run_index (int): run index, used for file name.
        dataset_path (str): Path to the dataset to be created.
        pop_diversity_metrics (list[PopDiversityMetric]): List of population diversity metrics to calculate.
        indiv_diversity_metrics (list[IndivDiversityMetric]): List of individual diversity metrics to calculate.
        max_iters (Optional[int]): Individual optimization run stopping condition.
        max_evals (Optional[int]): Individual optimization run stopping condition.
        rng_seed (Optional[int]): Seed for the rng, provide for reproducible results.
    """
    task = Task(problem, max_iters=max_iters, max_evals=max_evals)

    single_run_data = SingleRunData(
        algorithm_name=algorithm.Name,
        algorithm_parameters=algorithm.get_parameters(),
        problem_name=problem.name(),
        max_iters=max_iters,
        max_evals=max_evals,
    )

    optimization(
        algorithm,
        task,
        single_run_data,
        pop_diversity_metrics,
        indiv_diversity_metrics,
        rng_seed,
    )

    # check if folder structure exists, if not create it
    path = os.path.join(dataset_path, algorithm.Name[0], problem.name())
    if os.path.exists(path) == False:
        Path(path).mkdir(parents=True, exist_ok=True)

    single_run_data.export_to_json(os.path.join(path, f"run_{run_index:05d}.json"))


def optimization_runner(
    algorithm: Algorithm,
    problem: Problem,
    runs: int,
    dataset_path: str,
    pop_diversity_metrics: list[PopDiversityMetric],
    indiv_diversity_metrics: list[IndivDiversityMetric],
    max_iters: int = np.inf,
    max_evals: int = np.inf,
    rng_seed: int = None,
    parallel_processing=False,
):
    r"""Optimization work splitter.

    Args:
        algorithm (Algorithm): Algorithm.
        problem (Problem): Optimization problem.
        runs (int): Number of runs to execute.
        dataset_path (str): Path to the dataset to be created.
        pop_diversity_metrics (list[PopDiversityMetric]): List of population diversity metrics to calculate.
        indiv_diversity_metrics (list[IndivDiversityMetric]): List of individual diversity metrics to calculate.
        max_iters (Optional[int]): Individual optimization run stopping condition.
        max_evals (Optional[int]): Individual optimization run stopping condition.
        rng_seed (Optional[int]): Seed for the rng, provide for reproducible results.
        parallel_processing (Optional[bool]): Execute optimization runs in parallel over multiple processes.
    """
    if parallel_processing:
        pool = []
        for r_idx in range(runs):
            p = multiprocessing.Process(
                target=optimization_worker,
                args=(
                    problem,
                    algorithm,
                    r_idx,
                    dataset_path,
                    pop_diversity_metrics,
                    indiv_diversity_metrics,
                    max_iters,
                    max_evals,
                    rng_seed,
                ),
            )
            p.start()
            pool.append(p)

        for p in pool:
            p.join()
    else:
        for r_idx in range(runs):
            optimization_worker(
                problem=problem,
                algorithm=algorithm,
                run_index=r_idx,
                dataset_path=dataset_path,
                pop_diversity_metrics=pop_diversity_metrics,
                indiv_diversity_metrics=indiv_diversity_metrics,
                max_iters=max_iters,
                max_evals=max_evals,
                rng_seed=rng_seed,
            )
