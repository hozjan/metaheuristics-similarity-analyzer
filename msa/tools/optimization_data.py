from abc import ABC, abstractmethod
from typing import Any
from types import FunctionType
import numpy as np
import numpy.typing as npt
import pandas as pd
import json
import sklearn.preprocessing
from json import JSONEncoder
from niapy.problems import Problem
from sklearn.decomposition import PCA
import math
from msa.util.helper import smape

__all__ = ["IndivDiversityMetric", "PopDiversityMetric", "PopulationData", "SingleRunData", "JsonEncoder"]


class JsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, PopulationData):
            return json.dumps(obj.__dict__, indent=4, cls=JsonEncoder)
        return JSONEncoder.default(self, obj)


class IndivDiversityMetric(ABC):
    r"""Class representing a individual diversity metric.
    """

    def __init__(self, *args, **kwargs):
        r"""Initialize individual diversity metric."""

    @abstractmethod
    def _evaluate(self, srd: "SingleRunData", *args, **kwargs):
        r"""Evaluate individual diversity using a diversity metric."""
        pass

    def evaluate(self, srd: "SingleRunData"):
        r"""Evaluate individual diversity using a diversity metric.

        Args:
            srd (SingleRunData): Single run data containing populations of all iterations produced by an algorithm.

        Returns:
            List[float]: diversities of the individuals.

        """
        return self._evaluate(srd)

    def abbreviation(self):
        """Get diversity metric abbreviation."""
        return self.__class__.__name__


class PopDiversityMetric(ABC):
    r"""Class representing a population diversity metric.
    """

    def __init__(self, *args, **kwargs):
        r"""Initialize population diversity metric."""

    @abstractmethod
    def _evaluate(self, popData: "PopulationData", *args, **kwargs):
        r"""Evaluate population diversity using a diversity metric."""
        pass

    def evaluate(self, popData: "PopulationData"):
        r"""Evaluate population diversity using a diversity metric.

        Args:
            popData (PopulationData): Population data containing a single population produced by an algorithm.

        Returns:
            float: diversity of the population.

        """
        return self._evaluate(popData)

    def abbreviation(self):
        """Get diversity metric abbreviation."""
        return self.__class__.__name__


class PopulationData:
    r"""Class for archiving population data. Contains population, diversity
    metrics etc.

    Attributes:
        population (Optional[numpy.ndarray]): Population.
            population_fitness (Optional[numpy.ndarray]): Population fitness.
            best_solution (Optional[numpy.ndarray]): Best solution in the
                population.
            best_fitness (Optional[float]): Fitness of the best solution in
                the population.
            metrics_values (Dict[str, np.ndarray]): Dictionary of population
                diversity metrics values.
    """

    def __init__(
        self,
        population: npt.NDArray | None = None,
        population_fitness: npt.NDArray | None = None,
        best_solution: npt.NDArray | None = None,
        best_fitness: float | None = None,
    ):
        r"""Archive the population data and calculate diversity metrics.

        Args:
            population (Optional[numpy.ndarray]): Population.
            population_fitness (Optional[numpy.ndarray]): Population fitness.
            best_solution (Optional[numpy.ndarray]): Best solution in the
                population.
            best_fitness (Optional[float]): Fitness of the best solution in
                the population.

        Raises:
            ValueError: Attribute `population` was not defined.
            ValueError: Attribute `population_fitness` was not defined.
        """
        self.population = population
        self.population_fitness = population_fitness
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.metrics_values: dict[str, npt.NDArray] = {}

    def calculate_metrics(self, metrics: list[PopDiversityMetric], problem: Problem):
        r"""Calculate population diversity metrics.

        Args:
            metrics (List[DiversityMetric]): List of metrics to calculate.
            problem (Problem): Optimization problem.
        """

        if self.population is None:
            raise ValueError("Attribute `population` was not defined and thus metrics can not be calculated.")

        if self.population_fitness is None:
            raise ValueError("Attribute `population_fitness` was not defined and thus metrics can not be calculated.")

        for metric in metrics:
            self.metrics_values[metric.abbreviation()] = metric.evaluate(self)

    def __iter__(self):
        yield self

    def get_population_or_empty(self):
        if self.population is None:
            return np.array()
        else:
            return self.population

    def get_population_fitness_or_empty(self):
        if self.population_fitness is None:
            return np.array()
        else:
            return self.population_fitness


class SingleRunData:
    r"""Class for archiving optimization run data.
    Contains list of population data through iterations, run details such as
    problem used, algorithm used etc.

    Attributes:
        algorithm_name (Optional[str]): Algorithm name.
        algorithm_parameters (Optional[Dict[str, Any]]): Algorithm
            parameters.
        problem_name (Optional[str]): Problem name.
        max_evals (Optional[int]): Maximum number of function evaluations.
        max_iters (Optional[int]): Maximum number of generations or iterations.
        rng_seed (Optional[int]): Seed of the random generator used for
            optimization.
        evals (int): number of evaluations used.
        populations (List[PopulationData]): list of populations recorded
            during the solving of the problem.
        best_fitness (float): Fitness of the best individual.
        best_solution (np.ndarray): Best individual.
        indiv_metrics (Dict[str, np.ndarray]): Dictionary of the individual
            diversity metrics values.
        pop_metrics (Dict[str, np.ndarray]): Dictionary of the population
            diversity metrics values.
    """

    def __init__(
        self,
        algorithm_name: list[str] | None = None,
        algorithm_parameters: dict[str, Any] | None = None,
        problem_name: str | None = None,
        max_evals: int | float = np.inf,
        max_iters: int | float = np.inf,
        rng_seed: int | None = None,
    ):
        r"""Archive the optimization data through iterations.

        Args:
            algorithm_name (Optional[str]): Algorithm name.
            algorithm_parameters (Optional[Dict[str, Any]]): Algorithm
                parameters.
            problem_name (Optional[str]): Problem name.
            max_evals (Optional[int]): Maximum number of function evaluations.
            max_iters (Optional[int]): Maximum number of generations or iterations.
            rng_seed (Optional[int]): Seed of the random generator used for
                optimization.
        """
        self.algorithm_name = algorithm_name
        self.algorithm_parameters = algorithm_parameters
        self.problem_name = problem_name
        self.max_evals = max_evals
        self.max_iters = max_iters
        self.rng_seed = rng_seed
        self.evals = 0
        self.populations: list[PopulationData] = []
        self.best_fitness: float | None = None
        self.best_solution: npt.NDArray | None = None
        self.indiv_metrics: dict[str, npt.NDArray[np.float64]] = {}
        self.pop_metrics: dict[str, npt.NDArray[np.float64]] = {}

    def add_population(self, population_data: PopulationData):
        r"""Add population to list.

        Args:
            population (PopulationData): Population of type PopulationData.
        """
        self.populations.append(population_data)
        self.best_fitness = population_data.best_fitness
        self.best_solution = population_data.best_solution

    def get_pop_diversity_metrics_values(
        self,
        metrics: list[PopDiversityMetric] | None = None,
        minmax_scale: bool = False,
        standard_scale: bool = False,
    ):
        r"""Get population diversity metrics values.

        Args:
            metrics (List[PopDiversityMetric]): List of metrics to return.
                Returns all metrics if None (by default).
            minmax_scale (Optional[bool]): Method returns min-max scaled values
                to range [0,1] if true and `standard_scale` is false.
            standard_scale (Optional[bool]): Method returns standardized metrics if true.

        Returns:
            pandas.DataFrame: Metrics values throughout the run
        """
        if len(self.pop_metrics) == 0:
            for idx, population in enumerate(self.populations):
                for metric_abbr in population.metrics_values:
                    if idx == 0:
                        self.pop_metrics[metric_abbr] = np.array([])
                    self.pop_metrics[metric_abbr] = np.append(
                        self.pop_metrics[metric_abbr], population.metrics_values[metric_abbr]
                    )

        if not (metrics is None):
            _pop_metrics = {}
            for metric in metrics:
                if metric.abbreviation() in self.pop_metrics:
                    _pop_metrics[metric.abbreviation()] = self.pop_metrics.get(metric.abbreviation())
        else:
            _pop_metrics = dict(self.pop_metrics)

        if len(_pop_metrics) != 0:
            if standard_scale:
                for metric_abbr in _pop_metrics:
                    scaler = sklearn.preprocessing.StandardScaler()
                    _pop_metrics[metric_abbr] = (
                        scaler.fit_transform(np.array(_pop_metrics[metric_abbr]).reshape((-1, 1)))
                        .reshape((-1))
                        .tolist()
                    )
            elif minmax_scale:
                for metric_abbr in _pop_metrics:
                    _pop_metrics[metric_abbr] = sklearn.preprocessing.minmax_scale(
                        _pop_metrics[metric_abbr], feature_range=(0, 1)
                    )

        return pd.DataFrame.from_dict(_pop_metrics)

    def get_indiv_diversity_metrics_values(
        self,
        metrics: list[IndivDiversityMetric] | None = None,
        minmax_scale: bool = False,
        standard_scale: bool = False,
    ):
        r"""Get individual diversity metrics values.

        Args:
            metrics (Optional[List[IndivDiversityMetric]]): List of metrics to return.
                Returns all metrics if None (by default).
            minmax_scale (Optional[bool]):  Method returns min-max scaled values to
                range [0,1] if true and `standard_scale` is false.
            standard_scale (Optional[bool]): Method returns standardized metrics if true.

        Returns:
            pandas.DataFrame: Metrics values throughout the run
        """

        if metrics is not None:
            _indiv_metrics = {}
            for metric in metrics:
                if metric.abbreviation() in self.indiv_metrics:
                    _indiv_metrics[metric.abbreviation()] = self.indiv_metrics.get(metric.abbreviation())
        else:
            _indiv_metrics = dict(self.indiv_metrics)

        if minmax_scale:
            for metric_value in _indiv_metrics.keys():
                _indiv_metrics[metric_value] = sklearn.preprocessing.minmax_scale(
                    _indiv_metrics[metric_value], feature_range=(0, 1)
                )
        if standard_scale and len(_indiv_metrics) != 0:
            for metric_value in _indiv_metrics:
                scaler = sklearn.preprocessing.StandardScaler()
                _indiv_metrics[metric_value] = (
                    scaler.fit_transform(np.array(_indiv_metrics[metric_value]).reshape((-1, 1))).reshape((-1)).tolist()
                )

        return pd.DataFrame.from_dict(_indiv_metrics)

    def get_feature_vector(self, standard_scale: bool = True, minmax_scale: bool = False):
        r"""Calculate feature vector composed of catenated PCA eigenvectors multiplied by square root of corresponding
            eigenvalues of diversity metrics.

        Args:
            standard_scale (Optional[bool]): Use standard scaled diversity metrics.
            min_max_scale (Optional[bool]): Use min-max scaled diversity metrics.

        Returns:
            features (numpy.ndarray[float]): Vector of concatenated PCA eigenvectors multiplied by square root of
                corresponding eigenvalues of diversity metrics.
        """

        indiv_metrics = self.get_indiv_diversity_metrics_values(
            standard_scale=standard_scale, minmax_scale=minmax_scale
        )
        pop_metrics = self.get_pop_diversity_metrics_values(standard_scale=standard_scale, minmax_scale=minmax_scale)

        indiv_components = []
        pop_components = []

        pca_indiv = PCA(svd_solver="full", random_state=0)
        pca_indiv.fit(indiv_metrics)
        for component, value in zip(pca_indiv.components_, pca_indiv.explained_variance_):
            indiv_components.extend(component * math.sqrt(value))

        pca_pop = PCA(svd_solver="full", random_state=0)
        pca_pop.fit(pop_metrics)
        for component, value in zip(pca_pop.components_, pca_pop.explained_variance_):
            pop_components.extend(component * math.sqrt(value))

        return np.nan_to_num(np.concatenate((indiv_components, pop_components)))

    def get_diversity_metrics_similarity(self, second: "SingleRunData", get_raw_values: bool = False):
        r"""Calculate similarity based on 1-SMAPE between corresponding diversity metrics of two runs.

        Args:
            second (SingleRunData): SingleRunData object for diversity metrics comparison.
            get_raw_values (Optional[bool]): Returns an array of 1-SMAPE values if true.

        Returns:
            similarity (float | numpy.ndarray[float]): mean 1-SMAPE value or array of 1-SMAPE
                values if get_raw_values is true.
        """
        first_im = self.get_indiv_diversity_metrics_values().to_numpy().transpose()
        first_pm = self.get_pop_diversity_metrics_values().to_numpy().transpose()

        second_im = second.get_indiv_diversity_metrics_values().to_numpy().transpose()
        second_pm = second.get_pop_diversity_metrics_values().to_numpy().transpose()

        smape_values = []
        for fpm, spm in zip(first_pm, second_pm):
            smape_values.append(smape(fpm, spm))

        for fim, sim in zip(first_im, second_im):
            smape_values.append(smape(fim, sim))

        if get_raw_values:
            return np.array(smape_values)
        else:
            return np.mean(smape_values)

    def calculate_indiv_diversity_metrics(self, metrics: list[IndivDiversityMetric]):
        r"""Calculate Individual diversity metrics.
        Call suggested after optimization task stopping condition reached
        or when all populations added to the populations list.

        Args:
            metrics (List[DiversityMetric]): List of metrics to calculate.
        """
        for metric in metrics:
            self.indiv_metrics[metric.abbreviation()] = metric.evaluate(self)

    def get_best_fitness_values(self, normalize: bool = False):
        r"""Get array of best fitness values of all populations through the run.

        Returns:
            numpy.ndarray: Best fitness values throughout the run
        """
        fitness_values = np.array([])
        for p in self.populations:
            if p.best_fitness is not None:
                fitness_values = np.append(fitness_values, p.best_fitness)

        if normalize:
            fitness_values = sklearn.preprocessing.minmax_scale(fitness_values, feature_range=(0, 1))

        return fitness_values

    def export_to_json(self, filename, keep_pop_data: bool = True, keep_diversity_metrics: bool = True):
        r"""Export to json file.

        Args:
            filename (str): Filename of the output file. File extension .json has to be included.
            keep_pop_data (Optional[bool]): If false clear population solutions and fitness values in order to save
                space. Does not clear diversity metrics.
            keep_diversity_metrics (Optional[bool]): If false clear diversity metrics to further save space. Has no
                effect if keep_pop_data is true (true by default).
        """

        if not keep_pop_data:
            if keep_diversity_metrics:
                self.get_pop_diversity_metrics_values()
            else:
                self.indiv_metrics = {}
                self.pop_metrics = {}
            self.populations = []

        if self.algorithm_parameters is not None:
            for k, v in self.algorithm_parameters.items():
                if isinstance(v, FunctionType):
                    self.algorithm_parameters[k] = v.__name__

        json_object = json.dumps(self.__dict__, indent=4, cls=JsonEncoder)

        with open(filename, "w") as outfile:
            outfile.write(json_object)

    @staticmethod
    def import_from_json(filename: str):
        r"""Import data from the json file and create new class instance.

        Args:
            filename (str): Filename of the input file. File extension .json has to be included.

        Returns:
            (SingleRunData): instance of the `SingleRunData` object deserialized from .json file.

        Raises:
            FileNotFoundError: File not found.
            BaseException: File could not be loaded.
        """
        try:
            with open(filename) as file:
                data_dict = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} not found.")
        except Exception:
            raise BaseException(f"File {filename} could not be loaded.")

        single_run = SingleRunData(
            algorithm_name=data_dict["algorithm_name"],
            algorithm_parameters=data_dict["algorithm_parameters"],
            problem_name=data_dict["problem_name"],
            max_evals=data_dict["max_evals"],
            max_iters=data_dict["max_iters"],
            rng_seed=data_dict["rng_seed"],
        )

        single_run.evals = data_dict["evals"]
        single_run.indiv_metrics = data_dict["indiv_metrics"]
        single_run.pop_metrics = data_dict["pop_metrics"]
        single_run.best_fitness = data_dict["best_fitness"]
        single_run.best_solution = data_dict["best_solution"]
        single_run.populations.clear()
        if data_dict["populations"] is None or len(data_dict["populations"]) == 0:
            return single_run

        for pop in data_dict["populations"]:
            pop_dict = json.loads(pop)
            pop_data = PopulationData(
                population=pop_dict["population"],
                population_fitness=pop_dict["population_fitness"],
                best_solution=pop_dict["best_solution"],
                best_fitness=pop_dict["best_fitness"],
            )
            pop_data.metrics_values = pop_dict["metrics_values"]
            single_run.populations.append(pop_data)

        return single_run
