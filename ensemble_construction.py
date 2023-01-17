# -*- coding: utf-8 -*-
from json import tool
import random, util, os, sys
import numpy as np
from deap import base, creator, tools

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from nested_dichotomies import NestedDichotomie
from genotype import DistanceMatrix
from util import BinaryTreeNode, Ensemble, DecisionStump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import time, openml
from pathlib import Path

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

#takes output of nsga-II and seperates first rank from others
#then orders first rank by fitness
#only works for positive fitness values
def order_first_rank(individuals):
    if(not individuals):
        raise ValueError("No individuals in list")
    first_rank = [individuals[0]]
    pointer = 1
    while(pointer < len(individuals)):
        current_ind = individuals[pointer]
        position = 0
        while(position < len(first_rank) and first_rank[position].fitness.values[0] * first_rank[position].fitness.weights[0]  >= current_ind.fitness.values[0] * current_ind.fitness.weights[0]):
            position += 1
        if(position > 0 and first_rank[position -1].fitness.values[1] * first_rank[position -1].fitness.weights[1] >= current_ind.fitness.values[1] * current_ind.fitness.weights[1] and current_ind.fitness != first_rank[position -1].fitness):
            break
        first_rank.insert(position, current_ind)
        pointer += 1
    other_ranks = individuals[pointer :]
    return first_rank, other_ranks
    


def conda(X, y, base_learner_class):
    
    N_VALID_SPLITS = 5
    VALID_SIZE = 0.1
    
    N_POP_ND = 16
    N_POP_ENS = 20
    N_GEN = 200 #  200 #should be multiple of RESET_INTERVAL
    CX_PB_ND, MUT_PB_ND = 0.9, 1
    MUTATE_ETA_ND = 30
    CX_PB_ENS, MUT_PB_ENS = 0.7, 1
    
    JOIN_ENS_TOURNAMENT_SIZE = 2
    OFFSPRING_TOURNAMENT_SIZE = 2
    N_ENS_ASSIGNING_SUPPORT = N_POP_ENS
    
    EXPECTED_INIT_ENSEMBLE_SIZE = 64
    ENS_SIZE = 10
    N_TIMES_MORE_ENS_GENS = 5
    RESET_INTERVAL = 5 
    TIMEOUT = 420*5
    classes = unique_labels(y)
    n_classes = len(classes)
    X_trains, X_valids, y_trains, y_valids = [], [], [], []

    sss = StratifiedShuffleSplit(n_splits=N_VALID_SPLITS, test_size=VALID_SIZE)
    try:
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_trains.append(X_train)
            X_valids.append(X_valid)
            y_trains.append(y_train)
            y_valids.append(y_valid)

    except:
        X_trains, X_valids, y_trains, y_valids = [], [], [], []
        for i in range(N_VALID_SPLITS):
            X_train, X_valid , y_train, y_valid = train_test_split(X, y, test_size = VALID_SIZE)
            X_trains.append(X_train)
            X_valids.append(X_valid)
            y_trains.append(y_train)
            y_valids.append(y_valid)
    
    creator.create("ND_Fitness", base.Fitness, weights = (1.0, 1.0))
    creator.create("ND_Individual", list, fitness = creator.ND_Fitness, gen = 0, val_predictions = list, support = 0)
    creator.create("Ens_Fitness", base.Fitness, weights = (1.0, ))
    creator.create("Ens_Individual", list, fitness = creator.Ens_Fitness, gen = 0, val_predictions = list)
    
    def init_ens_population(n, nds):
        return [creator.Ens_Individual([random.choice(nds) for j in range(ENS_SIZE)]) for i in range(n)]
    
    def evaluate_nd(individual):
        if(individual.val_predictions != []):
            raise ValueError("ND Individual already contains predictions")
        
        dist_matr = DistanceMatrix(classes, individual)
        tree = dist_matr.build_tree()
        nd = NestedDichotomie(base_learner_class)
        
        accuracies = []
        for i in range(len(y_valids)):
            X_train, X_test, y_train, y_test = X_trains[i], X_valids[i], y_trains[i], y_valids[i]
            #todo: leave some X_train out
            nd = nd.fit(X_train, y_train, tree)
            valid_pred = nd.predict(X_test)
            individual.val_predictions.append(valid_pred)
            valid_accuracy = accuracy_score(y_test, valid_pred)
            accuracies.append(valid_accuracy)
        
        accuracies.sort()
        accuracies.pop(-1)
        accuracies.pop(0)
        accuracy = sum(accuracies)/len(accuracies)
        individual.phenotype = nd 
        return accuracy, individual.support
        
    def evaluate_ens(individual):
        if(individual.val_predictions != []):
            raise ValueError("Ens Individual already contains predictions")

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
            individual.val_predictions.append(pred)
        scores.sort()
        scores.pop(-1)
        scores.pop(0)
        accuracy = sum(scores)/len(scores)
        return accuracy, 
    
    def mate_ens(child1, child2):
        if(len(child1) ==1 or len(child2) ==1):
            i1 = random.randint(0, len(child1)-1)
            i2 = random.randint(0, len(child2)-1)
            temp = child1[i1]
            child1[i1] = child2[i2]
            child2[i2] = temp
        else:
            tools.cxTwoPoint(child1, child2)

    
    def mutate_ens(individual, nd_population):
        i = random.randint(0, len(individual) - 1)
        new_nd = toolbox.join_ens_tournament(nd_population, 1)[0]
        individual[i] = new_nd
        
    def genetic_operation(parameters, operator, probability):
        if(not isinstance(parameters, tuple)):
            raise ValueError("Pass Individual as tuple")
        applied = False
        if(random.random() < probability):
            applied = True
            operator(*parameters)
            for ind in parameters:
                try:
                    del ind.fitness.values
                except AttributeError:
                    pass
        return applied
                  
    def evaluate_population(fitness_func, pop):        
        for ind in pop:
            ind.fitness.values = fitness_func(ind)
    
    #currently not active
    #veraltet: support wird über attribut support assigned und nicht in die fitness eingetragen
    #todo: testen
    def assign_weighted_support_single(ens):
        support_weights_all_splits = []
        for i_valid_set in range(N_VALID_SPLITS):
            support_weights = []
            ens_predictions = ens.val_predictions[i_valid_set]
            for nd in ens:
                nd_predictions = nd.val_predictions[i_valid_set]
                diverging_predictions = nd_predictions != ens_predictions
                nd_correct_predictions = nd_predictions == y_valids[i_valid_set]
                nd_correct_predictions = nd_correct_predictions * 1 #convert bolean to int
                nd_correct_predictions[nd_correct_predictions == 0] = -1
                support_weights.append(sum(nd_correct_predictions[diverging_predictions]))
            support_weights_all_splits.append(support_weights)
        support_weights_all_splits = np.apply_along_axis(sum, 0, support_weights_all_splits)
        total_weights = sum(support_weights_all_splits)
        if(total_weights):
            for i in range(len(ens)):
                ens[i].fitness.values = (ens[i].fitness.values[0], ens[i].fitness.values[1] + (support_weights_all_splits[i] / total_weights))
    
    def assign_support_single(ens):
        for nd in ens:
            nd.support += 1
    
    def reset_support(nd_population):
        for ind in nd_population:
            ind.support = 0

    def evaluate_nd_support(nd):
        return (nd.fitness.values[0], nd.support)
    
    def count_nd_support(ens_population, nd_population):
        reset_support(nd_population)
        for i in range(N_ENS_ASSIGNING_SUPPORT):
            assign_support_single(ens_population[i])

    nd_ind_size = int((n_classes * (n_classes - 1)) / 2)
    gene_mut_prob = 1/nd_ind_size

    toolbox = base.Toolbox()
    toolbox.register("nd_individual", tools.initRepeat, creator.ND_Individual, random.random, nd_ind_size)
    toolbox.register("nd_population", tools.initRepeat, list, toolbox.nd_individual)
    toolbox.register("ens_population", init_ens_population)
    
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("offspring", tools.selTournament, tournsize = OFFSPRING_TOURNAMENT_SIZE)
            
    toolbox.register("mate_nd", tools.cxOnePoint)
    toolbox.register("mutate_nd", tools.mutPolynomialBounded, eta = MUTATE_ETA_ND, low = 0, up = 1, indpb = gene_mut_prob)
    toolbox.register("evaluate_nd", evaluate_nd)
    
    toolbox.register("mate_ens", mate_ens)
    toolbox.register("mutate_ens", mutate_ens)
    toolbox.register("evaluate_ens", evaluate_ens)
    toolbox.register("join_ens_tournament", tools.selTournament, tournsize = JOIN_ENS_TOURNAMENT_SIZE)
    
    toolbox.register("genetic_operation", genetic_operation)
    toolbox.register("evaluate_population", evaluate_population)
    toolbox.register("count_nd_support", count_nd_support)
    toolbox.register("evaluate_nd_support", evaluate_nd_support)

    nd_population = toolbox.nd_population(N_POP_ND)
    ens_population = toolbox.ens_population(N_POP_ENS, nd_population)
            
    toolbox.evaluate_population(toolbox.evaluate_nd, nd_population)
    toolbox.evaluate_population(toolbox.evaluate_ens, ens_population)

    start_time = time.time()
    top_ens_individual_age = 0 
    top_ens_individuals = []
    top_nd_individuals = []
    
    for g in range(N_GEN):
        
        generation_start_time = time.time()
        
        #reset
        if(g and g % RESET_INTERVAL == 0):
            nd_population[1:] = toolbox.nd_population(N_POP_ND-1)
            toolbox.evaluate_population(toolbox.evaluate_nd, nd_population[1:])
            
            ens_population_elite = [ens_population[0]]
            pointer_ens = 1
            while(pointer_ens < len(ens_population) and ens_population[pointer_ens].fitness.values[0] == ens_population_elite[0].fitness.values[0]):
                ens_population_elite.append(ens_population[pointer_ens])
                pointer_ens += 1
            ens_population[:] = ens_population_elite + toolbox.ens_population(N_POP_ENS - len(ens_population_elite), nd_population)
            #test
            if(len(ens_population) != N_POP_ENS):
                raise RuntimeError("ENS Population size shouldnt change")
            toolbox.evaluate_population(toolbox.evaluate_ens, ens_population[len(ens_population_elite):])
            
        
        #ND GENERATION
        # Select the offspring
        nd_offspring = toolbox.offspring(nd_population, N_POP_ND)
        # Clone the selected individuals
        nd_offspring = list(map(toolbox.clone, nd_offspring))
        # Apply crossover and mutation to the nd_offspring
        for child1, child2 in zip(nd_offspring[::2], nd_offspring[1::2]):
            genetic_operation((child1, child2), toolbox.mate_nd, CX_PB_ND)                
            genetic_operation((child1,), toolbox.mutate_nd, MUT_PB_ND)
            genetic_operation((child2,), toolbox.mutate_nd, MUT_PB_ND)
            #children copy their parents support 
            for child in (child1, child2):
                if(not child.fitness.valid):
                    child.gen = g + 1
                    child.val_predictions = []
                    child.fitness.values = toolbox.evaluate_nd(child)
        new_elite, rest = order_first_rank(toolbox.select(nd_population+nd_offspring, N_POP_ND))
        nd_population[:] = new_elite + rest

        #ENS LOOP
        for i in range(N_TIMES_MORE_ENS_GENS):
            ens_offspring = toolbox.offspring(ens_population, N_POP_ENS)
            ens_offspring = list(map(toolbox.clone, ens_offspring))

            for child1, child2 in zip(ens_offspring[::2], ens_offspring[1::2]):
                genetic_operation((child1, child2), toolbox.mate_ens, CX_PB_ENS)                
                genetic_operation((child1, nd_population), toolbox.mutate_ens, MUT_PB_ENS)
                genetic_operation((child2, nd_population), toolbox.mutate_ens, MUT_PB_ENS)
                for child in (child1, child2):
                    if(not child.fitness.valid):
                        child.gen = g+1
                        child.val_predictions = []
                        child.fitness.values = toolbox.evaluate_ens(child)
            ens_population[:] = toolbox.select(ens_population+ens_offspring, N_POP_ENS)
        toolbox.count_nd_support(ens_population, nd_population)
        toolbox.evaluate_population(toolbox.evaluate_nd_support, nd_population)
        new_elite, rest = order_first_rank(toolbox.select(nd_population, N_POP_ND))
        nd_population[:] = new_elite + rest


        top_nd_individuals.append(nd_population[0])
        top_ens_individuals.append(ens_population[0])



        if(time.time() - start_time > TIMEOUT):
            print("Evolutionary run timed out after {g+1} generations")
            generation_end_time = time.time()
            print(f"generation took {generation_end_time-generation_start_time}s")
            break
        
        if(len(top_ens_individuals) > 1 and top_ens_individuals[-2] != top_ens_individuals[-1]):
            top_ens_individual_age = 0
        else:
            top_ens_individual_age += 1
            
        if(top_ens_individual_age == 15):
            print("Top individual has not changed in 15 generations - stopping the run")
            break
        
        generation_end_time = time.time()


    del creator.ND_Individual
    del creator.ND_Fitness
    del creator.Ens_Individual
    del creator.Ens_Fitness

        
    return nd_population, ens_population, top_nd_individuals, top_ens_individuals
        

def simple_ndea(X, y, base_learner_class):
    
    N_VALID_SPLITS = 5
    VALID_SIZE = 0.1
    
    N_POP_ND = 16
    N_GEN = 200 #  200
    CX_PB_ND, MUT_PB_ND = 0.9, 1
    MUTATE_ETA_ND = 30
    
    OFFSPRING_TOURNAMENT_SIZE = 2
    
    RESET_INTERVAL = 5 
    
    classes = unique_labels(y)
    n_classes = len(classes)
    
    creator.create("ND_Fitness", base.Fitness, weights = (1.0,))
    creator.create("ND_Individual", list, fitness = creator.ND_Fitness, gen = 0)
    
    def evaluate_nd(individual):

        X_trains, X_valids, y_trains, y_valids = [], [], [], []
        sss = StratifiedShuffleSplit(n_splits=N_VALID_SPLITS, test_size=VALID_SIZE)
        try:
            for train_index, test_index in sss.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                X_trains.append(X_train)
                X_valids.append(X_valid)
                y_trains.append(y_train)
                y_valids.append(y_valid)

        except:
            X_trains, X_valids, y_trains, y_valids = [], [], [], []
            for i in range(N_VALID_SPLITS):
                X_train, X_valid , y_train, y_valid = train_test_split(X, y, test_size = VALID_SIZE)
                X_trains.append(X_train)
                X_valids.append(X_valid)
                y_trains.append(y_train)
                y_valids.append(y_valid) 
        dist_matr = DistanceMatrix(classes, individual)
        tree = dist_matr.build_tree()
        nd = NestedDichotomie(base_learner_class)
    
        
        accuracies = []
        for i in range(N_VALID_SPLITS):
            X_train, X_valid , y_train, y_valid = X_trains[i], X_valids[i], y_trains[i], y_valids[i]
            nd = nd.fit(X_train, y_train, tree)
            valid_pred = nd.predict(X_valid)
            valid_accuracy = accuracy_score(y_valid, valid_pred)
            accuracies.append(valid_accuracy)
        #trim accuracies - leave out best and worst
        accuracies.sort()
        accuracies.pop(-1)
        accuracies.pop(0)
        accuracy = sum(accuracies)/len(accuracies)
        individual.tree = tree
        return accuracy,
        
    def genetic_operation(parameters, operator, probability):
        if(not isinstance(parameters, tuple)):
            raise ValueError("Pass Individual/Parameters as tuple")
        applied = False
        if(random.random() < probability):
            applied = True
            operator(*parameters)
            for ind in parameters:
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
    
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("offspring", tools.selTournament, tournsize = OFFSPRING_TOURNAMENT_SIZE)
            
    toolbox.register("mate_nd", tools.cxOnePoint)
    toolbox.register("mutate_nd", tools.mutPolynomialBounded, eta = MUTATE_ETA_ND, low = 0, up = 1, indpb = gene_mut_prob)
    toolbox.register("evaluate_nd", evaluate_nd)
        
    toolbox.register("genetic_operation", genetic_operation)
    toolbox.register("evaluate_population", evaluate_population)

    start_time = time.time()
    nd_population = toolbox.nd_population(N_POP_ND)
    toolbox.evaluate_population(toolbox.evaluate_nd, nd_population)
    top_nd_individuals = []
    top_nd_individual_age = 0

    for g in range(N_GEN):
        
        #ND EVOLUTIONARY LOOP
        
        generation_start_time = time.time()
        #reset
        if(g and g % RESET_INTERVAL == 0):
            nd_population[1:] = toolbox.nd_population(N_POP_ND-1)
            toolbox.evaluate_population(toolbox.evaluate_nd, nd_population[1:])
        
        # Select the offspring
        nd_offspring = toolbox.offspring(nd_population, N_POP_ND)
        # Clone the selected individuals
        nd_offspring = list(map(toolbox.clone, nd_offspring))

        # Apply crossover and mutation to the nd_offspring
        for child1, child2 in zip(nd_offspring[::2], nd_offspring[1::2]):
            genetic_operation((child1, child2), toolbox.mate_nd, CX_PB_ND)                
            genetic_operation((child1,), toolbox.mutate_nd, MUT_PB_ND)
            genetic_operation((child2,), toolbox.mutate_nd, MUT_PB_ND)

            for child in (child1, child2):
                if(not child.fitness.valid):
                    child.gen = g + 1
                    child.fitness.values = toolbox.evaluate_nd(child)
        nd_population[:] = toolbox.select(nd_population + nd_offspring, N_POP_ND)
        top_nd_individuals.append(nd_population[0])
        
        #stopping criteria
        if(time.time() - start_time > 420):
            print("Evolutionary run timed out after {g+1} generations")
            generation_end_time = time.time()
            print(f"generation took {generation_end_time-generation_start_time}s")
            break
        
        if(len(top_nd_individuals) > 1 and top_nd_individuals[-2] != top_nd_individuals[-1]):
            top_nd_individual_age = 0
        else:
            top_nd_individual_age += 1
            
        if(top_nd_individual_age == 15):
            print("Top individual has not changed in 15 generations - stopping the run")
            break
        
        generation_end_time = time.time()

        
    del creator.ND_Individual
    del creator.ND_Fitness
    return nd_population, top_nd_individuals


def display_population(pop):
    print(f" number of individuals: {len(pop)} \n")
    
    distinct_pop = []
    for i in range(len(pop)):
        unseen = True
        for j in range(i):
            if(pop[i] == pop[j]):
                unseen = False
        if(unseen):
            distinct_pop.append(pop[i])
    
    print(f" number of distinct individuals: {len(distinct_pop)} \n")
    #print(f" distinct individuals: {distinct_pop} \n")
    print(f"fitness: {[ind.fitness.values for ind in distinct_pop]} \n")
    print(f"generations: {[ind.gen for ind in pop]} \n")
    
    try:
        if(isinstance(pop[0], creator.Ens_Individual)):
            print(f"Ensemble sizes: {[len(ind) for ind in pop]}")
    except IndexError:
        print("Empy population exception")
        

def log_result(experiment_id, method, base_learner, task, fold, accuracy, train_accuracy, ensemble_size, duration, other_results = None, folder = "results-3"):
    log_message = "experiment id: {}, method: {}, base_learner: {} task: {}, fold: {}, accuracy: {}, train_accuracy: {}, ensemble_size: {}, duration: {}, other_results = {}\n".format(experiment_id, method, base_learner, task, fold, accuracy, train_accuracy, ensemble_size, duration, other_results)
    dirname = os.path.dirname(__file__)
    results_dirname = os.path.join(dirname, folder + "/" + str(experiment_id))
    Path(results_dirname).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(results_dirname, 'results.txt')
    with open(filename, "a") as f:
        f.write(log_message)


@ignore_warnings(category=ConvergenceWarning)
def single_experiment(method, task_id, fold_id, base_learner, experiment_id=-1):
    task = openml.tasks.get_task(task_id)
    dataset = openml.datasets.get_dataset(task.dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    #preprocessing
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    categorical_features = list(X.columns[categorical_indicator])
    numeric_features = list(X.columns[~np.array(categorical_indicator)])
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = OneHotEncoder(sparse = False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    X = preprocessor.fit_transform(X)    
    start_time = time.time()
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold_id, sample=0)
    X_train, y_train, X_test, y_test = X[train_indices], y[train_indices], X[test_indices], y[test_indices]
    other_results = []
    if(method == "ndea"):
        ENSEMBLE_SIZE = 10 #  TO 10 DONT FORGET
        top_ensemble = Ensemble()
        for j in range(ENSEMBLE_SIZE):
            final_population, top_nd_individuals_over_time =simple_ndea(X_train, y_train, base_learner)
            top_individual = top_nd_individuals_over_time[-1]
            top_individual_blueprint = top_individual.tree
            train_accuracy=top_individual.fitness.values[0]
            new_nd = NestedDichotomie(base_learner)
            new_nd = new_nd.fit(X_train, y_train, top_individual_blueprint)
            top_ensemble.append(new_nd)
            top_nd_individuals_fitness_over_time = [nd.fitness.values[0] for nd in top_nd_individuals_over_time]
            other_results.append("top nd individuals per generation: " + str(top_nd_individuals_fitness_over_time))
    elif(method == "conda"):
        final_nd_population, final_ens_population, top_nd_individuals_over_time, top_ens_individuals_over_time = conda(X_train, y_train, base_learner)
        top_ens_individual = top_ens_individuals_over_time[-1]
        train_accuracy = top_ens_individual.fitness.values[0]
        top_ensemble = Ensemble([nd_genotype.phenotype for nd_genotype in top_ens_individual])
        top_ensemble.refit(X_train,y_train)

        for i in range(len(final_ens_population)):
            ens_individual = final_ens_population[i]
            ens_train_accuracy = ens_individual.fitness.values[0]
            ensemble = Ensemble([nd_genotype.phenotype for nd_genotype in ens_individual])
            ensemble.refit(X_train,y_train)
            ensemble_predictions = ensemble.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
            other_results.append(("size: "+ str(len(ensemble)), "test accuracy:" +str(ensemble_accuracy), "train accuracy: " + str(ens_train_accuracy)))
        top_nd_individuals_fitness_over_time = [(nd.fitness.values[0], nd.fitness.values[1]) for nd in top_nd_individuals_over_time]        
        top_ens_individuals_fitness_over_time = [(ens.fitness.values[0],) for ens in top_ens_individuals_over_time]
        other_results.append("top nd individuals per generation: " + str(top_nd_individuals_fitness_over_time))
        other_results.append("top ens individuals per generation: " + str(top_ens_individuals_fitness_over_time))
    else:
        raise ValueError("Method can be 'ndea' or 'conda'")
    predictions = top_ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    ensemble_size = len(top_ensemble)
    duration = time.time() - start_time
    if(base_learner == DecisionTreeClassifier):
        base_learner_name = "Decision Tree"
    elif(base_learner == DecisionStump):
        base_learner_name = "Decision Stump"
    else:
        print("Unrecognized base learner")
        base_learner_name = str(base_learner)
    log_result(experiment_id, method, base_learner_name, task_id, fold_id, accuracy, train_accuracy, ensemble_size, duration, other_results)


#simple_ndea_experiment(6)
#simple_ndea_experiment(40)

def main():
    tasks = [2, 9, 40, 146204, 18, 9964, 41, 3022, 145681, 7]
    base_learners = [DecisionTreeClassifier, DecisionStump, LogisticRegression]
    methods = ["ndea", "conda"]
    experiment_configurations = [(method, task_id, fold_id, base_learner) for method in methods for task_id in tasks for fold_id in range(10) for base_learner in base_learners]
    
    experiment_id = int(sys.argv[1]) 
    if(experiment_id < 0 or experiment_id > len(experiment_configurations)-1):
        raise ValueError("Illegal experiment ID")
    
    
    single_experiment(*experiment_configurations[experiment_id], experiment_id)
    


def test(id = None):
    tasks = [2, 9, 40, 146204, 18, 9964, 41, 3022, 145681, 7]
    base_learners = [DecisionTreeClassifier, DecisionStump] 
    methods = ["ndea", "conda"]
    experiment_configurations = [(method, task_id, fold_id, base_learner) for method in methods for task_id in tasks for fold_id in range(10) for base_learner in base_learners]
    
    if(id is not None):
        print(experiment_configurations[id])
        single_experiment(*experiment_configurations[id], id)
    else:
        for experiment_id in range(len(experiment_configurations)):
            print(experiment_configurations[experiment_id])
            single_experiment(*experiment_configurations[experiment_id], experiment_id)
        
#test(399)
main() 




