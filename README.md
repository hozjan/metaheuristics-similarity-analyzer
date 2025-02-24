# MSA - Metaheuristics Similarity Analyzer

This repository contains the source code of the experiments in the paper ***Measuring the similarity of metaheuristic search strategies with machine learning models***. MSA provides an alternative way to compare and analyze metaheuristic-search strategies with the help of machine learning.

## Usage
To use MSA for similarity analysis we first have to define the gene spaces which will be used by the genetic algorithm. First key of the gene space dictionary must correspond with the class name of the the algorithm which must be implemented in the [NiaPy](https://github.com/NiaOrg/NiaPy?tab=readme-ov-file) micro-framework. In this case we chose `BatAlgorithm` and `ParticleSwarmAlgorithm`.

```python
PSA_gene_spaces = {
    "ParticleSwarmAlgorithm": {
        "c1": {"low": 0.01, "high": 2.5, "step": 0.01},
        "c2": {"low": 0.01, "high": 2.5, "step": 0.01},
        "w": {"low": 0.0, "high": 1.0, "step": 0.01},
    }
}
BA_gene_spaces = {
    "BatAlgorithm": {
        "loudness": {"low": 0.01, "high": 1.0, "step": 0.01},
        "pulse_rate": {"low": 0.01, "high": 1.0, "step": 0.01},
        "alpha": {"low": 0.9, "high": 1.0, "step": 0.001},
        "gamma": {"low": 0.0, "high": 1.0, "step": 0.01},
    }
}
```
We also have to chose diversity metrics which will be used as the basis of the analysis.

```python
from msa.util.pop_diversity_metrics import PopDiversityMetric
from msa.util.indiv_diversity_metrics import IndivDiversityMetric

POP_DIVERSITY_METRICS = [
    PopDiversityMetric.PDC,
    PopDiversityMetric.FDC,
    PopDiversityMetric.PFSD,
    PopDiversityMetric.PFM,
]
INDIV_DIVERSITY_METRICS = [
    IndivDiversityMetric.IDT,
    IndivDiversityMetric.ISI,
    IndivDiversityMetric.IFM,
    IndivDiversityMetric.IFIQR,
]
```

In the next step we have to instantiate the `MetaGA` class which uses the `GA` class from the [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython) library. At this point we decide which of the algorithms will be analyzed and which will be the "reference" or "target" algorithm. The gene space of the analyzed algorithm gets assigned to the `gene_spaces` argument of the `MetaGA`.

```python
from msa.tools.meta_ga import MetaGA, MetaGAFitnessFunction
from msa.problems.schwefel import Schwefel

meta_ga = MetaGA(
    fitness_function_type=MetaGAFitnessFunction.TARGET_PERFORMANCE_SIMILARITY,
    ga_generations=20,
    ga_solutions_per_pop=15,
    ga_percent_parents_mating=60,
    ga_parent_selection_type="tournament",
    ga_k_tournament=2,
    ga_crossover_type="uniform",
    ga_mutation_type="random",
    ga_crossover_probability=0.9,
    ga_mutation_num_genes=1,
    ga_keep_elitism=1,
    gene_spaces=BA_gene_spaces,
    pop_size=30,
    max_evals=10000,
    num_runs=30,
    problem=Schwefel(20),
    pop_diversity_metrics=POP_DIVERSITY_METRICS,
    indiv_diversity_metrics=INDIV_DIVERSITY_METRICS,
)
```

In the last step we have to instantiate the `MetaheuristicSimilarityAnalyzer` class and pass it the configured `MetaGA` instance and the gene space of the target algorithm. Then we simply call the `run_similarity_analysis` method to start the analysis.

```python
msa = MetaheuristicSimilarityAnalyzer(meta_ga=meta_ga, target_gene_space=PSA_gene_spaces)

msa.run_similarity_analysis(
    get_info=True,
    generate_dataset=True,
    export=True,
    calculate_similarity_metrics=True,
)
```

After the analysis we can choose to export results of the analysis as a .pdf and/or .tex file with `export_results_to_latex` method ar access them trough the `MetaheuristicSimilarityAnalyzer` class instance.

```python
msa.export_results_to_latex(generate_pdf=True)
```

## This project depends on
### [NiaPy](https://github.com/NiaOrg/NiaPy?tab=readme-ov-file) Python microframework
### [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython) Python library
