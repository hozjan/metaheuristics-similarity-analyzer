import numpy as np
from msa.tools.optimization_data import PopulationData, SingleRunData
from niapy.problems import Problem


def GenerateSingleRunData(pop_size: int, dimension: int, iterations: int, problem: Problem):
    srd = SingleRunData()
    for i in range(0, iterations):
        pop_data = GeneratePopulationData(pop_size, dimension, problem)
        srd.add_population(pop_data)

    return srd


def GeneratePopulationData(pop_size: int, dimension: int, problem: Problem):
    population = np.zeros(shape=(pop_size, dimension))
    pop_fitness = []
    for p in population:
        pop_fitness.append(problem.evaluate(p))

    population_data = PopulationData(population, np.array(pop_fitness))

    return population_data
