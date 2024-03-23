from typing import Dict, Any, List
from types import FunctionType
import numpy as np
import json
from json import JSONEncoder
from collections import namedtuple
from niapy.problems import Problem

from util.diversity_metrics import PDC, DiversityMetric

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

    def calculate_metrics(self, metrics: List[DiversityMetric], problem: Problem = None):
        r"""Calculate diversity metrics.

        Args:
            metrics (List[DiversityMetric]): List of metrics to calculate.
            problem (Problem): Optimization problem.
        """
        for metric in metrics:
            match metric:
                case DiversityMetric.PDC:
                    self.metrics_values[metric.value] = PDC(self.population, problem)


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

    def add_population(self, population_data: PopulationData):
        r"""Add population to list.

        Args:
            population (PopulationData): Population of type PopulationData.
        """

        self.populations.append(population_data)

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

    def import_from_json(self, filename):
        r"""Import data from the json file.

        Args:
            filename (str): Filename of the input file.
        """

        file = open(filename)
        data_dict = json.load(file)
        self.algorithm_name = data_dict["algorithm_name"]
        self.algorithm_parameters = data_dict["algorithm_parameters"]
        self.problem_name = data_dict["problem_name"]
        self.max_evals = data_dict["max_evals"]
        self.max_iters = data_dict["max_iters"]
        self.populations.clear()
        for pop in data_dict["populations"]:
            pop_dict = json.loads(pop)
            pop_data = PopulationData(
                population=pop_dict["population"],
                population_fitness=pop_dict["population_fitness"],
            )
            pop_data.metrics_values = pop_dict["metrics_values"]
            self.populations.append(pop_data)
