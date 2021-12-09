# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:14:21 2021

@author: luish_000
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 06:36:48 2021

@author: luish_000
"""

import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

#Problem parameter
NB_CITIES = 5

recorrido = [[0, 7, 9, 8, 20],
       [7, 0, 10, 4, 11],
       [9, 10, 0, 15, 5],
       [8, 4, 15, 0, 17],
       [20, 11, 5, 17, 0]
       ]

def evalPosicion(individual):
    suma = 0
    start = individual[0]
    for i in range(1, len(individual)):
        end = individual[i]
        suma += recorrido[start][end]
        start = end
    suma += recorrido[individual[0]][end]
    return suma,



creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

#Since there is only one queen per line, 
#individual are represented by a permutation
toolbox = base.Toolbox()
toolbox.register("permutation", random.sample, range(NB_CITIES), NB_CITIES)

#Structure initializers
#An individual is a list that represents the position of each queen.
#Only the line is stored, the column is the index of the number in the list.
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalPosicion)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0/NB_CITIES)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(seed=0):
    random.seed(seed)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats,
                        halloffame=hof, verbose=True)

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
    print('*****RECORRIDO*****')
    print('Individual: ', pop[0])
    print('Suma de recorrido: ', evalPosicion(pop[0]))