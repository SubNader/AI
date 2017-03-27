import array
import random
import json
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# Fetch data from the JSON file
with open("data.json", "r") as tsp_data_file:
    distance_data = json.load(tsp_data_file)
distance_map = distance_data["DistanceMatrix"]
trip_size = distance_data["TourSize"]

# Create classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Register methods
toolbox.register("indices", random.sample, range(trip_size), trip_size)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Run the TSP GA
def run_tsp(individual):
    distance = distance_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return distance,

# Register the crossover,mutation, selection and evolution methods
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", run_tsp)


def main():
    # The population of the cities
    the_population = toolbox.population(n=trip_size)

    # Best of all generated paths
    hall_of_fame = tools.HallOfFame(1)

    # Create stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Average", numpy.mean)
    stats.register("Standard Deviation", numpy.std)
    stats.register("Minimum", numpy.min)
    stats.register("Maximum", numpy.max)

    # Run the GA
    algorithms.eaSimple(the_population, toolbox, 0.5, 0.25, 100, stats=stats, halloffame=hall_of_fame)
    return the_population, stats, hall_of_fame


if __name__ == "__main__":
    main()