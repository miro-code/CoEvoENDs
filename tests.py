from json import tool
import random
from deap import tools, creator, base
from nested_dichotomies import NestedDichotomie
from genotype import DistanceMatrix

def test_init_ens_population_concept(n):
    p_add = 1- (1 / n)
    result = []
    for i in range(10000):
        result.append(1)
        while(random.random() < p_add):
            result[i] += 1
    return sum(result)/len(result)



def test_ea():
    creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)    
    toolbox = base.Toolbox()
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", lambda ind : (ind[0] , ind[1]))
    population = [creator.Individual([(10-i)/10, 1 + i]) for i in range(10)]
    population.insert(5, creator.Individual([0.9, 0.9]))
    
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    print("original: " + str(population))
    population = toolbox.select(population, 20)
    print("nsga2: " + str(population))
    first_rank, rest = ndea.order_first_rank(population)
    print("first rank: " + str(first_rank))
    print(" rest: " + str(rest))

    toolbox.register("evaluate_single", lambda ind : (ind[0] ,))
    creator.create("SingleFitness", base.Fitness, weights=(1.0, ))
    creator.create("SingleIndividual", list, fitness=creator.SingleFitness)
    population = [creator.SingleIndividual([i]) for i in range(10)]
    population.insert(0, creator.SingleIndividual([5]))
    for ind in population:
        ind.fitness.values = toolbox.evaluate_single(ind)

    print(population)
    population = tools.selNSGA2(population, 3)
    print(population)




def test_single():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)    
    toolbox = base.Toolbox()
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("select_tournament", tools.selTournament)
    toolbox.register("evaluate", lambda ind : (ind[0] , ))
    population = [creator.Individual([i]) for i in range(10)]
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    print(population)


def random_tree(classes):
    n_classes = len(classes)
    nd_ind_size = int((n_classes * (n_classes - 1)) / 2)
    individual = [random.random() for i in range(nd_ind_size)]
    dist_matr = DistanceMatrix(classes, individual)
    tree = dist_matr.build_tree()
    return tree



