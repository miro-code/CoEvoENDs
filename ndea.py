# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:11:35 2022

@author: pro
"""

import random, util
from deap import base, creator, tools

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from nested_dichotomies import NestedDichotomie
from genotype import DistanceMatrix
from util import BinaryTreeNode



def mccv(classifier, X, y, cv, valid_size):
    results = []
    for i in range(cv):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = valid_size) 
        classifier = classifier.fit(X_train, y_train)
        valid_pred = classifier.predict(X_test)
        print(y_test)
        print(valid_pred == y_test)
        valid_accuracy = accuracy_score(y_test, valid_pred)
        results.append(valid_accuracy)
    return results
        

def ndea(X, y):
    classes = unique_labels(y)
    n_classes = len(classes)
    creator.create("FitnessMax", base.Fitness, weights = (1.0, ))
    creator.create("Individual", list, fitness = creator.FitnessMax, gen = 0)
    
    ind_size = int((n_classes * (n_classes - 1)) / 2)
    
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n = ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        dist_matr = DistanceMatrix(classes, individual)
        tree = dist_matr.build_tree()
        nd = NestedDichotomie(LogisticRegression, tree)
        scores = mccv(nd, X, y, 5, 0.1)
        #scores = cross_val_score(nd, X, y, cv = 5, scoring = "accuracy")
        result = sum(scores)/len(scores)
        return result, 
        
            
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta = 0.1, low = 0, up = 1, indpb = 1/ind_size)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("offspring", tools.selTournament, tournsize = 2)
    toolbox.register("evaluate", evaluate)
    
    N_POP = 5
    pop = toolbox.population(n=N_POP)
    CXPB, MUTPB, N_GEN = 0.5, 0.2, 2

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(N_GEN):
        print("\n\n\nGENERATION " + str(g) + "\n")
        # Select the next generation individuals
        offspring = toolbox.offspring(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.gen = g + 1
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = toolbox.select(pop + offspring, N_POP)
    return pop

#todo: implement stopping criteria

def endea(X, y):

    N_SPLITS = 5
    VALID_SIZE = 0.1
    
    EXPECTED_INIT_ENSEMBLE_SIZE = 2
    
    N_POP_ND = 5
    N_POP_ENS = 10
    N_GEN = 3
    CXPB_ND, MUTPB_ND = 0.5, 0.2
    MUTATE_ETA_ND = 100
    CXPB_ENS, MUTPB_ENS = 0.5, 0.2

    classes = unique_labels(y)
    n_classes = len(classes)
    X_trains, X_valids, y_trains, y_valids = [], [], [], []

    for i in range(N_SPLITS):
        X_train, X_valid , y_train, y_valid = train_test_split(X, y, test_size = VALID_SIZE)
        X_trains.append(X_train)
        X_valids.append(X_valid)
        y_trains.append(y_train)
        y_valids.append(y_valid)
    
    creator.create("ND_Fitness", base.Fitness, weights = (1.0, 1.0))
    creator.create("ND_Individual", list, fitness = creator.ND_Fitness, gen = 0, val_predictions = [], support = 0)
    creator.create("Ensemble_Fitness", base.Fitness, weights = (1.0, -1.0))
    creator.create("Ensemble_Individual", list, fitness = creator.Ensemble_Fitness, gen = 0)
    
    def init_ens_population(n, nds):
        #größe der individuen geometrisch verteilt
        p_add = 1 / EXPECTED_INIT_ENSEMBLE_SIZE
        result = []
        for i in range(N_POP_ENS):
            result.append(creator.Ensemble_Individual([random.choice(nds)]))
            while(random.random() < p_add):
                result[i].append(random.choice(nds))
        return result
    
    def evaluate_nd(individual):
        if(individual.val_predictions != []):
            raise ValueError("Individual already contains predictions")
        
        dist_matr = DistanceMatrix(classes, individual)
        tree = dist_matr.build_tree()
        nd = NestedDichotomie(LogisticRegression, tree)
        
        scores = []
        for i in range(len(y_valids)):
            X_train, X_test, y_train, y_test = X_trains[i], X_valids[i], y_trains[i], y_valids[i]
            nd = nd.fit(X_train, y_train)
            valid_pred = nd.predict(X_test)
            individual.val_predictions.append(valid_pred)
            valid_accuracy = accuracy_score(y_test, valid_pred)
            scores.append(valid_accuracy)
        accuracy = sum(scores)/len(scores)
        return accuracy, individual.support
        
    def evaluate_ens(individual):
        
        scores = []
        for i in range(len(y_valids)):
            predictions = []
            for nd in individual:
                try:
                    predictions.append(nd.val_predictions[i])
                except IndexError:
                    raise ValueError("ND INDIVIDUAL LACKS VALIDATION PREDICTIONS")
            predictions = np.array(predictions)
            pred = np.apply_along_axis(util.max_bincount_random_tiebreak, 0, predictions)
            scores.append(accuracy_score(y_valids[i], pred))
        accuracy = sum(scores)/len(scores)
        return accuracy, len(individual)
    
    def mutate_ens(individual, nd_population):
        pass

    def genetic_operation(individuals, operator, probability):
        if(not isinstance(individuals, tuple)):
            raise ValueError("Pass Individual as tuple")
        applied = False
        if(random.random() < probability):
            applied = True
            operator(*individuals)
            for ind in individuals:
                try:
                    del ind.fitness.values
                except AttributeError:
                    pass
        return applied
                  
    def evaluate_population(fitness_func, pop):        
        for ind in pop:
            ind.fitness.values = fitness_func(ind)
    
    nd_ind_size = int((n_classes * (n_classes - 1)) / 2)
    gene_mut_prob = 1/nd_ind_size

    toolbox = base.Toolbox()
    toolbox.register("nd_individual", tools.initRepeat, creator.ND_Individual, random.random, nd_ind_size)
    toolbox.register("nd_population", tools.initRepeat, list, toolbox.nd_individual)
    toolbox.register("ens_population", init_ens_population)
    
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("offspring", tools.selTournament, tournsize = 2)
            
    toolbox.register("mate_nd", tools.cxOnePoint)
    toolbox.register("mutate_nd", tools.mutPolynomialBounded, eta = MUTATE_ETA_ND, low = 0, up = 1, indpb = gene_mut_prob)
    toolbox.register("evaluate_nd", evaluate_nd)
    
    toolbox.register("mate_ens", tools.cxTwoPoint)
    toolbox.register("mutate_ens", mutate_ens)
    toolbox.register("evaluate_ens", evaluate_ens)

    toolbox.register("genetic_operation", genetic_operation)
    toolbox.register("evaluate_population", evaluate_population)

    nd_population = toolbox.nd_population(N_POP_ND)
    ens_population = toolbox.ens_population(N_POP_ENS, nd_population)
            
    toolbox.evaluate_population(toolbox.evaluate_nd, nd_population)
    toolbox.evaluate_population(toolbox.evaluate_ens, ens_population)

    for g in range(N_GEN):
        
        #ND EVOLUTIONARY LOOP

        # Select the offspring
        offspring = toolbox.offspring(nd_population, len(nd_population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation to the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            
            applied_cx = genetic_operator((child1, child2), toolbox.mate_nd, CXPB_ND)
            if(applied_cx):
                child1.val_predictions = []
                child2.val_predictions = []

            applied_mut_child1 = genetic_operator((child1), toolbox.mutate_nd, MUTPB_ND)
            if(applied_mut_child1):
                child1.val_predictions = []
            
            applied_mut_child2 = genetic_operator((child2), toolbox.mutate_nd, MUTPB_ND)
            if(applied_mut_child2):
                child2.val_predictions = []

            for child in (child1, child2):
                if(not child.fitness.valid):
                    child.gen = g + 1
                    child.fitness.values = toolbox.evaluate_nd(child)

        # The population is entirely replaced by the offspring
        pop[:] = toolbox.select(pop + offspring, N_POP)
    return pop


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv("C:\\Users\\pro\\Downloads\\iris.data")

outcomes = df.iloc[:,4]

le = LabelEncoder()
outcomes = le.fit_transform(outcomes)

features = df.drop(df.columns[[4]], axis = 1)

#X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size = 0.3) 

# =============================================================================
# population = ndea(features, outcomes)
# population.sort(key = lambda x: x.fitness.values[0])
# print(population)
# fitnesses = [i.fitness for i in population]
# print(fitnesses)
# generations = [i.gen for i in population]
# print(generations)
# =============================================================================

population = endea(features, outcomes)

