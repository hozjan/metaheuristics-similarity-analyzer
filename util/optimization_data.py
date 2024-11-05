from typing import Dict, Any, List
from types import FunctionType
import numpy as np
import pandas as pd
import json
import sklearn.preprocessing
from json import JSONEncoder
from niapy.util.distances import euclidean
from niapy.problems import Problem
from sklearn.decomposition import PCA
from scipy import spatial
import math
from util.helper import smape

from util.pop_diversity_metrics import (
    PDC,
    PED,
    PMD,
    AAD,
    PDI,
    FDC,
    PFSD,
    PFM,
    PopDiversityMetric,
)
from util.indiv_diversity_metrics import (
    IDT,
    ISI,
    IFM,
    IFIQR,
    IndivDiversityMetric,
)

__all__ = ["PopulationData", "SingleRunData", "JsonEncoder"]


class JsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, PopulationData):
            return json.dumps(obj.__dict__, indent=4, cls=JsonEncoder)
        return JSONEncoder.default(self, obj)


class PopulationData:
    r"""Class for archiving population data. Contains population, diversity metrics etc."""

    def __init__(
        self,
        population=None,
        population_fitness=None,
        best_solution=None,
        best_fitness=None,
    ):
        r"""Archive the population data and calculate diversity metrics.

        Args:
            population (Optional[numpy.ndarray]): Population.
            population_fitness (Optional[numpy.ndarray]): Population fitness.
            best_solution (Optional[numpy.ndarray]): Best solution in the population.
            best_fitness (Optional[float]): Fitness of the best solution in the population.
        """
        self.population = population
        self.population_fitness = population_fitness
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.metrics_values = {}

    def calculate_metrics(
        self, metrics: List[PopDiversityMetric], problem: Problem = None
    ):
        r"""Calculate diversity metrics.

        Args:
            metrics (List[DiversityMetric]): List of metrics to calculate.
            problem (Problem): Optimization problem.
        """
        for metric in metrics:
            match metric:
                case PopDiversityMetric.PDC:
                    self.metrics_values[metric.value] = PDC(self.population, problem)
                case PopDiversityMetric.PED:
                    self.metrics_values[metric.value] = PED(self.population)
                case PopDiversityMetric.PMD:
                    self.metrics_values[metric.value] = PMD(self.population)
                case PopDiversityMetric.AAD:
                    self.metrics_values[metric.value] = AAD(self.population)
                case PopDiversityMetric.PDI:
                    self.metrics_values[metric.value] = PDI(self.population, problem)
                case PopDiversityMetric.PFSD:
                    self.metrics_values[metric.value] = PFSD(self.population_fitness)
                case PopDiversityMetric.PFM:
                    self.metrics_values[metric.value] = PFM(self.population_fitness)
                case PopDiversityMetric.FDC:
                    self.metrics_values[metric.value] = FDC(
                        self.population, self.population_fitness, problem
                    )


class SingleRunData:
    r"""Class for archiving optimization run data.
    Contains list of population data through iterations, run details such as problem used, algorithm used etc.
    """

    def __init__(
        self,
        algorithm_name: str = None,
        algorithm_parameters: Dict[str, Any] = None,
        problem_name: str = None,
        max_evals: int = np.inf,
        max_iters: int = np.inf,
        rng_seed: int = None,
    ):
        r"""Archive the optimization data through iterations.

        Args:
            algorithm_name (Optional[str]): Algorithm name.
            algorithm_parameters (Optional[Dict[str, Any]]): Algorithm parameters.
            problem_name (Optional[str]): Problem name.
            max_evals (Optional[int]): Number of function evaluations.
            max_iters (Optional[int]): Number of generations or iterations.
            rng_seed (Optional[int]): Seed of the random generator used for optimization.
        """
        self.algorithm_name = algorithm_name
        self.algorithm_parameters = algorithm_parameters
        self.problem_name = problem_name
        self.max_evals = max_evals
        self.max_iters = max_iters
        self.rng_seed = rng_seed
        self.evals = 0
        self.populations = []
        self.best_fitness = None
        self.best_solution = None
        self.indiv_metrics = {}
        self.pop_metrics = {}

    def add_population(self, population_data: PopulationData):
        r"""Add population to list.

        Args:
            population (PopulationData): Population of type PopulationData.
        """
        self.populations.append(population_data)
        self.best_fitness = population_data.best_fitness
        self.best_solution = population_data.best_solution

    def get_pop_diversity_metrics_values(
        self, metrics: list[PopDiversityMetric] = None, normalize=False
    ):
        r"""Get population diversity metrics values.

        Args:
            metrics (List[DiversityMetric]): List of metrics to return. Returns all metrics if None (by default).
            normalize (Optional[bool]): Method returns normalized values if true.

        Returns:
            pandas.DataFrame: Metrics values throughout the run
        """
        if len(self.pop_metrics.keys()) == 0:
            for idx, population in enumerate(self.populations):
                for metric in population.metrics_values:
                    if idx == 0:
                        self.pop_metrics[metric] = []
                    self.pop_metrics[metric].append(population.metrics_values[metric])

        if not (metrics is None):
            _pop_metrics = {}
            for metric in metrics:
                if metric.value in self.pop_metrics:
                    _pop_metrics[metric.value] = self.pop_metrics.get(metric.value)
        else:
            _pop_metrics = dict(self.pop_metrics)

        if normalize and len(_pop_metrics) != 0:
            for metric in _pop_metrics:
                _pop_metrics[metric] = sklearn.preprocessing.minmax_scale(
                    _pop_metrics[metric], feature_range=(0, 1)
                )

        return pd.DataFrame.from_dict(_pop_metrics)

    def get_indiv_diversity_metrics_values(self, normalize=False):
        r"""Get individual diversity metrics values.

        Args:
            normalize (Optional[bool]): Method returns normalized values if true.

        Returns:
            pandas.DataFrame: Metrics values throughout the run
        """
        _indiv_metrics = dict(self.indiv_metrics)

        if normalize:
            for metric in _indiv_metrics:
                _indiv_metrics[metric] = sklearn.preprocessing.minmax_scale(
                    _indiv_metrics[metric], feature_range=(0, 1)
                )

        return pd.DataFrame.from_dict(_indiv_metrics)
    
    def get_combined_feature_vector(self, normalize=True):
        r"""Calculate feature vector composed of PCA eigenvectors and eigenvalues of diversity metrics.

        Args:
            normalize (Optional[bool]): Take normalized metrics.

        Returns:
            features (numpy.ndarray[float]): Vector of combined PCA eigenvectors and eigenvalues of diversity metrics.
        """
        indiv_metrics = self.get_indiv_diversity_metrics_values(normalize=normalize)
        pop_metrics = self.get_pop_diversity_metrics_values(normalize=normalize)
        
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

        return np.nan_to_num(
            np.concatenate(
                (indiv_components, pop_components)
            )
        )
    
    def get_combined_feature_vector_from_pairwise_normalized_metrics(self, second:"SingleRunData"):
        r"""Calculate feature vector composed of PCA eigenvectors and eigenvalues of diversity metrics,
        by normalizing diversity metrics pairwise between runs.

        Returns:
            features (numpy.ndarray[float]): Vector of combined PCA eigenvectors and eigenvalues of diversity metrics.
        """
        first_im = self.get_indiv_diversity_metrics_values(normalize=False).to_numpy().transpose()
        first_pm = self.get_pop_diversity_metrics_values(normalize=False).to_numpy().transpose()

        second_im = second.get_indiv_diversity_metrics_values(normalize=False).to_numpy().transpose()
        second_pm = second.get_pop_diversity_metrics_values(normalize=False).to_numpy().transpose()

        for idx in range(len(first_pm)):
            min_value = min(np.min(first_pm[idx]), np.min(second_pm[idx]))
            max_value = max(np.max(first_pm[idx]), np.max(second_pm[idx]))

            first_pm[idx] = (first_pm[idx] - min_value)/(max_value - min_value)
            second_pm[idx] = (second_pm[idx] - min_value)/(max_value - min_value)

        for idx in range(len(first_im)):
            min_value = min(np.min(first_im[idx]), np.min(second_im[idx]))
            max_value = max(np.max(first_im[idx]), np.max(second_im[idx]))

            first_im[idx] = (first_im[idx] - min_value)/(max_value - min_value)
            second_im[idx] = (second_im[idx] - min_value)/(max_value - min_value)

        first_im = first_im.transpose()
        first_pm = first_pm.transpose()
        second_im = second_im.transpose()
        second_pm = second_pm.transpose()
        
        first_indiv_components = []
        first_pop_components = []
        second_indiv_components = []
        second_pop_components = []

        pca_indiv = PCA(svd_solver="full", random_state=0)
        pca_indiv.fit(first_im)
        for component, value in zip(pca_indiv.components_, pca_indiv.explained_variance_):
            first_indiv_components.extend(component * math.sqrt(value))

        pca_pop = PCA(svd_solver="full", random_state=0)
        pca_pop.fit(first_pm)
        for component, value in zip(pca_pop.components_, pca_pop.explained_variance_):
            first_pop_components.extend(component * math.sqrt(value))

        first_fv = np.nan_to_num(
            np.concatenate(
                (first_indiv_components, first_pop_components)
            )
        )

        pca_indiv = PCA(svd_solver="full", random_state=0)
        pca_indiv.fit(second_im)
        for component, value in zip(pca_indiv.components_, pca_indiv.explained_variance_):
            second_indiv_components.extend(component * math.sqrt(value))

        pca_pop = PCA(svd_solver="full", random_state=0)
        pca_pop.fit(second_pm)
        for component, value in zip(pca_pop.components_, pca_pop.explained_variance_):
            second_pop_components.extend(component * math.sqrt(value))

        second_fv = np.nan_to_num(
            np.concatenate(
                (second_indiv_components, second_pop_components)
            )
        )


        return first_fv, second_fv
    
    def get_diversity_metrics_similarity(self, second: "SingleRunData", get_raw_values=False):
        r"""Calculate similarity based on 1-SMAPE between corresponding diversity metrics of two runs.

        Args:
            second (SingleRunData): SingleRunData object for diversity metrics comparison.
            get_raw_values (Optional[bool]): Returns an array of 1-SMAPE values if true.

        Returns:
            similarity (float | numpy.ndarray[float]): mean 1-SMAPE value or array of 1-SMAPE values if get_raw_values is true.
        """
        first_im = self.get_indiv_diversity_metrics_values(normalize=False).to_numpy().transpose()
        first_pm = self.get_pop_diversity_metrics_values(normalize=False).to_numpy().transpose()

        second_im = second.get_indiv_diversity_metrics_values(normalize=False).to_numpy().transpose()
        second_pm = second.get_pop_diversity_metrics_values(normalize=False).to_numpy().transpose()
        
        smape_values = []
        for fpm, spm in zip(first_pm, second_pm):
            smape_values.append(smape(fpm, spm))

        for fim, sim in zip(first_im, second_im):
            smape_values.append(smape(fim, sim))

        if get_raw_values:
            return np.array(smape_values)
        else:
            return np.mean(smape_values)


    def calculate_indiv_diversity_metrics(self, metrics):
        r"""Calculate Individual diversity metrics.
        Call suggested after optimization task stopping condition reached
        or when all populations added to the populations list.

        Args:
            metrics (List[DiversityMetric]): List of metrics to calculate.
        """

        if IndivDiversityMetric.IDT in metrics and IndivDiversityMetric.ISI in metrics:
            (
                self.indiv_metrics[IndivDiversityMetric.ISI.value],
                self.indiv_metrics[IndivDiversityMetric.IDT.value],
            ) = ISI(
                self.populations,
                self.algorithm_parameters["population_size"],
                return_idt=True,
            )

        for metric in metrics:
            if metric.value in self.indiv_metrics:
                continue
            match metric:
                case IndivDiversityMetric.IDT:
                    self.indiv_metrics[IndivDiversityMetric.IDT.value] = IDT(
                        self.populations, self.algorithm_parameters["population_size"]
                    )
                case IndivDiversityMetric.ISI:
                    self.indiv_metrics[IndivDiversityMetric.ISI.value] = ISI(
                        self.populations, self.algorithm_parameters["population_size"]
                    )
                case IndivDiversityMetric.IFM:
                    self.indiv_metrics[IndivDiversityMetric.IFM.value] = IFM(
                        self.populations, self.algorithm_parameters["population_size"]
                    )
                case IndivDiversityMetric.IFIQR:
                    self.indiv_metrics[IndivDiversityMetric.IFIQR.value] = IFIQR(
                        self.populations
                    )

    def get_best_fitness_values(self, normalize=False):
        r"""Get array of best fitness values of all populations through the run.

        Returns:
            numpy.ndarray: Best fitness values throughout the run
        """
        fitness_values = np.array([])
        for p in self.populations:
            fitness_values = np.append(fitness_values, p.best_fitness)

        if normalize:
            fitness_values = sklearn.preprocessing.minmax_scale(
                fitness_values, feature_range=(0, 1)
            )

        return fitness_values

    def export_to_json(self, filename, keep_pop_data=True, keep_diversity_metrics=True):
        r"""Export to json file.

        Args:
            filename (str): Filename of the output file. File extension .json has to be included.
            keep_pop_data (Optional[bool]): If false clear population solutions and fitness values in order to save space. Does not clear diversity metrics.
            keep_diversity_metrics (Optional[bool]): If false clear diversity metrics to further save space. Has no effect if keep_pop_data is true (true by default).
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
    def import_from_json(filename):
        r"""Import data from the json file and create new class instance.

        Args:
            filename (str): Filename of the input file. File extension .json has to be included.
        """
        try:
            with open(filename) as file:
                data_dict = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename}.json not found.")
        except:
            raise BaseException(f"File {filename}.json could not be loaded.")

        single_run = SingleRunData()
        single_run.algorithm_name = data_dict["algorithm_name"]
        single_run.algorithm_parameters = data_dict["algorithm_parameters"]
        single_run.problem_name = data_dict["problem_name"]
        single_run.max_evals = data_dict["max_evals"]
        single_run.max_iters = data_dict["max_iters"]
        single_run.rng_seed = data_dict["rng_seed"]
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
