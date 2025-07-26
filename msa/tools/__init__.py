"""Module containing useful tools of the MSA"""

from msa.tools.meta_ga import MetaGA
from msa.tools.metaheuristics_similarity_analyzer import MetaheuristicsSimilarityAnalyzer as MSA
from msa.tools.ml_tools import (
    svm_and_knn_classification,
)
from msa.tools.optimization_tools import optimization, optimization_worker, optimization_runner
from msa.tools.optimization_data import (
    IndivDiversityMetric,
    PopDiversityMetric,
    SingleRunData,
    PopulationData,
    JsonEncoder,
)

__all__ = [
    "MetaGA",
    "MSA",
    "svm_and_knn_classification",
    "optimization",
    "optimization_worker",
    "optimization_runner",
    "SingleRunData",
    "PopulationData",
    "JsonEncoder",
    "IndivDiversityMetric",
    "PopDiversityMetric",
]
