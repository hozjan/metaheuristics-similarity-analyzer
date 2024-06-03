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

from util.pop_diversity_metrics import (
    PDC,
    PED,
    PMD,
    AAD,
    PDI,
    PFSD,
    PFMea,
    PFMed,
    PopDiversityMetric,
)
from util.indiv_diversity_metrics import IDT, ISI, IFMea, IFMed, IndivDiversityMetric

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
                case PopDiversityMetric.PFMea:
                    self.metrics_values[metric.value] = PFMea(self.population_fitness)
                case PopDiversityMetric.PFMed:
                    self.metrics_values[metric.value] = PFMed(self.population_fitness)


class SingleRunData:
    r"""Class for archiving optimization run data.
    Contains list of population data through iterations, run details such as problem used, algorithm used etc.
    """

    def __init__(
        self,
        algorithm_name: str = None,
        algorithm_parameters: Dict[str, Any] = None,
        problem_name: str = None,
        max_evals=None,
        max_iters=None,
    ):
        r"""Archive the optimization data through iterations.

        Args:
            algorithm_name (Optional[str]): Algorithm name.
            algorithm_parameters (Optional[Dict[str, Any]]): Algorithm parameters.
            problem_name (Optional[str]): Problem name.
            max_evals (Optional[int]): Number of function evaluations.
            max_iters (Optional[int]): Number of generations or iterations.
        """
        self.algorithm_name = algorithm_name
        self.algorithm_parameters = algorithm_parameters
        self.problem_name = problem_name
        self.max_evals = max_evals
        self.max_iters = max_iters
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

    def get_pop_diversity_metrics_values(self, normalize=False):
        r"""Get population diversity metrics values.

        Args:
            normalize (bool): method returns normalized values if true.

        Returns:
            pandas.DataFrame: metrics values throughout the run
        """
        if len(self.pop_metrics.keys()) == 0:
            for idx, population in enumerate(self.populations):
                for metric in population.metrics_values:
                    if idx == 0:
                        self.pop_metrics[metric] = []
                    self.pop_metrics[metric].append(
                        population.metrics_values[metric]
                    )

        _pop_metrics = dict(self.pop_metrics)

        if normalize:
            for metric in _pop_metrics:
                _pop_metrics[metric][-1] = 0.0
                _pop_metrics[metric] = sklearn.preprocessing.minmax_scale(
                    _pop_metrics[metric], feature_range=(0, 1)
                )

        return pd.DataFrame.from_dict(_pop_metrics)

    def get_indiv_diversity_metrics_values(self, normalize=False):
        r"""Get individual diversity metrics values.

        Args:
            normalize (bool): method returns normalized values if true.

        Returns:
            pandas.DataFrame: metrics values throughout the run
        """
        _indiv_metrics = dict(self.indiv_metrics)

        if normalize:
            for metric in _indiv_metrics:
                _indiv_metrics[metric] = sklearn.preprocessing.minmax_scale(
                    _indiv_metrics[metric], feature_range=(0, 1)
                )

        return pd.DataFrame.from_dict(_indiv_metrics)

    def diversity_metrics_euclidean_distance(
        self, run: "SingleRunData", include_fitness_convergence=False, normalize=False
    ):
        r"""Calculate the sum of euclidean distances between corresponding diversity metrics of two individual runs.

        Args:
            run (SingleRunData): Single run data for comparison to the current instance.
            include_fitness_convergence (bool): Also include euclidean distance of the fitness convergence.
            normalize (bool): Method returns a sum of euclidean distances between normalized metrics if true.

        Returns:
            float: Sum of euclidean distances.
        """
        first_pdm = np.transpose(
            self.get_pop_diversity_metrics_values(normalize).to_numpy(), (1, 0)
        )
        second_pdm = np.transpose(
            run.get_pop_diversity_metrics_values(normalize).to_numpy(), (1, 0)
        )

        first_idm = self.get_indiv_diversity_metrics_values(normalize).to_numpy()
        second_idm = run.get_indiv_diversity_metrics_values(normalize).to_numpy()

        euclidean_sum = 0
        for first, second in zip(first_pdm, second_pdm):
            euclidean_sum += euclidean(first, second)

        f_pca = PCA(n_components=first_idm.shape[1])
        f_principal_components = f_pca.fit_transform(first_idm).flatten()
        s_pca = PCA(n_components=second_idm.shape[1])
        s_principal_components = s_pca.fit_transform(second_idm).flatten()
        euclidean_sum += euclidean(f_principal_components, s_principal_components)

        if include_fitness_convergence:
            first_fitness = self.get_best_fitness_values(normalize)
            second_fitness = run.get_best_fitness_values(normalize)
            euclidean_sum += euclidean(first_fitness, second_fitness)

        return euclidean_sum

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
                case IndivDiversityMetric.IFMea:
                    self.indiv_metrics[IndivDiversityMetric.IFMea.value] = IFMea(
                        self.populations, self.algorithm_parameters["population_size"]
                    )
                case IndivDiversityMetric.IFMed:
                    self.indiv_metrics[IndivDiversityMetric.IFMed.value] = IFMed(
                        self.populations
                    )

    def get_best_fitness_values(self, normalize=False):
        r"""Get array of best fitness values of all populations through the run.

        Returns:
            numpy.ndarray: best fitness values throughout the run
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
            filename (str): Filename of the output file.
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

        # Writing to sample.json
        with open(filename, "w") as outfile:
            outfile.write(json_object)

    @staticmethod
    def import_from_json(filename):
        r"""Import data from the json file and create new class instance.

        Args:
            filename (str): Filename of the input file.
        """
        single_run = SingleRunData()
        file = open(filename)
        data_dict = json.load(file)
        single_run.algorithm_name = data_dict["algorithm_name"]
        single_run.algorithm_parameters = data_dict["algorithm_parameters"]
        single_run.problem_name = data_dict["problem_name"]
        single_run.max_evals = data_dict["max_evals"]
        single_run.max_iters = data_dict["max_iters"]
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
