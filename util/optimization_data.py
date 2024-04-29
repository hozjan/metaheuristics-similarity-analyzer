from typing import Dict, Any, List
from types import FunctionType
import numpy as np
import pandas as pd
import json
import sklearn.preprocessing
from json import JSONEncoder
from collections import namedtuple
from niapy.problems import Problem

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

    def __init__(self, population=None, population_fitness=None):
        r"""Archive the population data and calculate diversity metrics.

        Args:
            population (Optional[numpy.ndarray]): Population.
            population_fitness (Optional[numpy.ndarray]): Population fitness.
        """
        self.population = population
        self.population_fitness = population_fitness
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
        self.indiv_metrics = {}

    def add_population(self, population_data: PopulationData):
        r"""Add population to list.

        Args:
            population (PopulationData): Population of type PopulationData.
        """
        self.populations.append(population_data)

    def get_pop_diversity_metrics_values(self, normalize=False):
        r"""Get population diversity metrics values.

        Args:
            normalize (bool): method returns normalized values if true.

        Returns:
            pandas.DataFrame: metrics values throughout the run
        """
        metrics = pd.DataFrame({})
        metrics_abbr = []
        metrics_values = []
        for idx, population in enumerate(self.populations):
            for metric in population.metrics_values:
                if idx == 0:
                    metrics_abbr.append(metric)
                    metrics_values.append([])
                metrics_values[metrics_abbr.index(metric)].append(
                    population.metrics_values[metric]
                )

        if normalize:
            metrics_values = sklearn.preprocessing.minmax_scale(
                metrics_values, feature_range=(0, 1), axis=1
            )

        for idx, metric in enumerate(metrics_abbr):
            metrics[metric] = metrics_values[idx]

        return metrics

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
            ) = ISI(self.populations, self.algorithm_parameters["population_size"], return_idt=True)

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

    def export_to_json(self, filename):
        r"""Export to json file.

        Args:
            filename (str): Filename of the output file.
        """
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
        single_run.populations.clear()
        for pop in data_dict["populations"]:
            pop_dict = json.loads(pop)
            pop_data = PopulationData(
                population=pop_dict["population"],
                population_fitness=pop_dict["population_fitness"],
            )
            pop_data.metrics_values = pop_dict["metrics_values"]
            single_run.populations.append(pop_data)

        return single_run
