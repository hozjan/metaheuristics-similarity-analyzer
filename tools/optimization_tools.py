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
from util.constants import RNG_SEED, DATASET_PATH

__all__ = ["optimization", "optimization_worker", "optimization_runner"]


def optimization(algorithm: Algorithm, task: Task, single_run_data: SingleRunData):
    r"""An adaptation of NiaPy Algorithm run method.

    Args:
        algorithm (Algorithm): Algorithm.
        task (Task): Task with pre configured parameters.
        single_run_data (SingleRunData): Instance for archiving optimization results
    """
    try:
        algorithm.callbacks.before_run()
        algorithm.rng = default_rng(seed=RNG_SEED)
        pop, fpop, params = algorithm.init_population(task)
        # reset seed to random
        algorithm.rng = default_rng()
        xb, fxb = algorithm.get_best(pop, fpop)
        while not task.stopping_condition():
            # save population data
            pop_data = PopulationData(
                population=np.array(pop), population_fitness=np.array(fpop)
            )
            pop_data.calculate_metrics(
                [
                    PopDiversityMetric.PDC,
                    PopDiversityMetric.PED,
                    PopDiversityMetric.PMD,
                    PopDiversityMetric.AAD,
                    PopDiversityMetric.PFSD,
                    PopDiversityMetric.PFMea,
                    PopDiversityMetric.PFMed,
                ],
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
        single_run_data.calculate_indiv_diversity_metrics(
            [
                IndivDiversityMetric.IDT,
                IndivDiversityMetric.ISI,
                IndivDiversityMetric.IFMea,
                IndivDiversityMetric.IFMed,
            ]
        )
        return xb, fxb * task.optimization_type.value
    except BaseException as e:
        if (
            threading.current_thread() is threading.main_thread()
            and multiprocessing.current_process().name == "MainProcess"
        ):
            raise e
        algorithm.exception = e
        return None, None


def optimization_worker(problem, algorithm, max_iter, run_index):
    r"""Single optimization run execution.

    Args:
        algorithm (Algorithm): Algorithm.
        problem (Problem): Optimization problem.
        max_iter (int): Optimization stopping condition.
        run_index (int): run index, used for file name.
    """
    task = Task(problem, max_iters=max_iter)

    single_run_data = SingleRunData(
        algorithm_name=algorithm.Name,
        algorithm_parameters=algorithm.get_parameters(),
        problem_name=problem.name(),
        max_iters=max_iter,
    )

    optimization(algorithm, task, single_run_data)

    # check if folder structure exists, if not create it
    path = os.path.join(DATASET_PATH, algorithm.Name[0], problem.name())
    if os.path.exists(path) == False:
        Path(path).mkdir(parents=True, exist_ok=True)

    single_run_data.export_to_json(os.path.join(path, f"run_{run_index:05d}.json"))


def optimization_runner(
    algorithm: Algorithm,
    problem: Problem,
    max_iter: int,
    runs: int,
    parallel_processing=False,
):
    r"""Optimization work splitter.

    Args:
        algorithm (Algorithm): Algorithm.
        problem (Problem): Optimization problem.
        max_iter (int): Optimization stopping condition.
        runs (int): Number of runs to execute.
        parallel_processing (Optional[bool]): Execute optimization runs in parallel over multiple processes.
    """
    if parallel_processing:
        pool = []
        for r in range(runs):
            p = multiprocessing.Process(
                target=optimization_worker,
                args=(
                    problem,
                    algorithm,
                    max_iter,
                    r,
                ),
            )
            p.start()
            pool.append(p)

        for p in pool:
            p.join()
    else:
        for r in range(runs):
            optimization_worker(problem, algorithm, max_iter, r)
