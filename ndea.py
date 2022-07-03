# -*- coding: utf-8 -*-

import random, util
import numpy as np
from deap import base, creator, tools

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from nested_dichotomies import NestedDichotomie
from genotype import DistanceMatrix
from util import BinaryTreeNode, Ensemble
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import time, openml

def endea(X, y):
    
    BASE_LEARNER_CLASS = LogisticRegression
    N_VALID_SPLITS = 5
    VALID_SIZE = 0.1
    
    N_POP_ND = 5
    N_POP_ENS = 20
    N_GEN = 5
    CX_PB_ND, MUT_PB_ND = 0.8, 0.8
    MUTATE_ETA_ND = 1
    CX_PB_ENS, MUT_PB_ENS = 0.8, 0.8
    
    MUT_APPEND_PB_ENS, MUT_REDUCE_PB_ENS, MUT_ALTER_PB_ENS = 0.7, 0.05, 0.07
    JOIN_ENS_TOURNAMENT_SIZE = 2
    OFFSPRING_TOURNAMENT_SIZE = 2
    N_SUPPORT_ASSIGNED = 2
    
    EXPECTED_INIT_ENSEMBLE_SIZE = 20
    N_TIMES_MORE_ENS_GENS = 1
    RESET_INTERVAL = 2
    
    classes = unique_labels(y)
    n_classes = len(classes)
    X_trains, X_valids, y_trains, y_valids = [], [], [], []

    for i in range(N_VALID_SPLITS):
        X_train, X_valid , y_train, y_valid = train_test_split(X, y, test_size = VALID_SIZE)
        X_trains.append(X_train)
        X_valids.append(X_valid)
        y_trains.append(y_train)
        y_valids.append(y_valid)
    
    creator.create("ND_Fitness", base.Fitness, weights = (1.0, 1.0))
    creator.create("ND_Individual", list, fitness = creator.ND_Fitness, gen = 0, val_predictions = list, support = 0)
    creator.create("Ens_Fitness", base.Fitness, weights = (1.0, -1.0))
    creator.create("Ens_Individual", list, fitness = creator.Ens_Fitness, gen = 0, val_predictions = list)
    
    def init_ens_population(n, nds):
        #größe der individuen geometrisch verteilt
        p_add = 1- (1 / EXPECTED_INIT_ENSEMBLE_SIZE)
        result = []
        for i in range(N_POP_ENS):
            result.append(creator.Ens_Individual([random.choice(nds)]))
            while(random.random() < p_add):
                result[i].append(random.choice(nds))
        return result
    
    def evaluate_nd(individual):
        if(individual.val_predictions != []):
            raise ValueError("ND Individual already contains predictions")
        
        dist_matr = DistanceMatrix(classes, individual)
        tree = dist_matr.build_tree()
        nd = NestedDichotomie(BASE_LEARNER_CLASS)
        
        scores = []
        for i in range(len(y_valids)):
            X_train, X_test, y_train, y_test = X_trains[i], X_valids[i], y_trains[i], y_valids[i]
            nd = nd.fit(X_train, y_train, tree)
            valid_pred = nd.predict(X_test)
            individual.val_predictions.append(valid_pred)
            valid_accuracy = accuracy_score(y_test, valid_pred)
            scores.append(valid_accuracy)
        accuracy = sum(scores)/len(scores)
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
        accuracy = sum(scores)/len(scores)
        return accuracy, len(individual)
    
    def mate_ens(child1, child2):
        if(len(child1) < 2 or len(child2) < 2):
            i1 = random.randint(0, len(child1)-1)
            i2 = random.randint(0, len(child2)-1)
            temp = child1[i1]
            child1[i1] = child2[i2]
            child2[i2] = temp
        else:
            tools.cxTwoPoint(child1, child2)

    
    def mutate_ens(individual, nd_population):
        if(len(individual) > 1 and random.random() < MUT_REDUCE_PB_ENS):
            i = random.randint(0, len(individual) - 1)
            individual.pop(i)
        if(random.random() < MUT_APPEND_PB_ENS):
            new_nd = toolbox.join_ens_tournament(nd_population, 1)[0] #tournSel retourns list of selected individuals
            individual.insert(random.randint(0, len(individual)), new_nd)
        if(random.random() < MUT_ALTER_PB_ENS):
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
            nd.fitness.values = (nd.fitness.values[0], nd.fitness.values[1] + 1)
    
    def reset_support(nd_population):
        for ind in nd_population:
            ind.fitness.values = (ind.fitness.values[0], 0.0)
    
    def assign_support_population(ens_population, nd_population):
        reset_support(nd_population)
        for i in range(N_SUPPORT_ASSIGNED):
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
    toolbox.register("assign_support", assign_support_population)

    nd_population = toolbox.nd_population(N_POP_ND)
    ens_population = toolbox.ens_population(N_POP_ENS, nd_population)
            
    toolbox.evaluate_population(toolbox.evaluate_nd, nd_population)
    toolbox.evaluate_population(toolbox.evaluate_ens, ens_population)
    top_ens_individual = None
    start_time = time.time()
    
    
    for g in range(N_GEN):
        
        #ND EVOLUTIONARY LOOP
        
        #reset
        if(g and g % RESET_INTERVAL == 0):
            nd_population[1:] = toolbox.nd_population(N_POP_ND)[1:]
            toolbox.evaluate_population(toolbox.evaluate_nd, nd_population)
        
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
                    child.val_predictions = []
                    child.fitness.values = toolbox.evaluate_nd(child)

        nd_population[:] = toolbox.select(nd_offspring + nd_population, N_POP_ND)

        #nd_population += nd_offspring #for debugging

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
            ens_population[:] = toolbox.select(ens_offspring + ens_population, N_POP_ENS)
        toolbox.assign_support(ens_population, nd_population)
        
        #stopping criteria
        if(time.time() - start_time < 420):
            print("Evolutionary run timed out")
            break
        
        if(top_ens_individual != ens_population[0]):
            top_ens_individual_age = 0
            top_ens_individual = ens_population[0]
        else:
            top_ens_individual_age += 1
            
        if(top_ens_individual_age == 15):
            print("Top ensemble has not changed in 15 generations - stopping the run")
            break
        
        
        
    return nd_population, ens_population
        

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
    print(f" distinct individuals: {distinct_pop} \n")
    print(f"fitness: {[ind.fitness.values for ind in distinct_pop]} \n")
    print(f"generations: {[ind.gen for ind in pop]} \n")
    
    try:
        if(isinstance(pop[0], creator.Ens_Individual)):
            print(f"Ensemble sizes: {[len(ind) for ind in pop]}")
    except IndexError:
        print("Empy population exception")
        

def simple_ndea(X, y, base_learner_class):
    
    N_VALID_SPLITS = 5
    VALID_SIZE = 0.1
    
    N_POP_ND = 5
    N_GEN = 1
    CX_PB_ND, MUT_PB_ND = 0.8, 0.8
    MUTATE_ETA_ND = 1
    
    OFFSPRING_TOURNAMENT_SIZE = 2
    
    RESET_INTERVAL = 5
    
    classes = unique_labels(y)
    n_classes = len(classes)
    
    creator.create("ND_Fitness", base.Fitness, weights = (1.0,))
    creator.create("ND_Individual", list, fitness = creator.ND_Fitness, gen = 0)
    
    
    def evaluate_nd(individual):
        dist_matr = DistanceMatrix(classes, individual)
        tree = dist_matr.build_tree()
        nd = NestedDichotomie(base_learner_class)
    
        accuracies = []
        for i in range(N_VALID_SPLITS):
            X_train, X_valid , y_train, y_valid = train_test_split(X, y, test_size = VALID_SIZE)
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
    top_nd_individual = None
    for g in range(N_GEN):
        
        #ND EVOLUTIONARY LOOP
        
        generation_start_time = time.time()
        #reset
        if(g and g % RESET_INTERVAL == 0):
            nd_population[1:] = toolbox.nd_population(N_POP_ND)[1:]
            toolbox.evaluate_population(toolbox.evaluate_nd, nd_population)
        
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

        nd_population[:] = toolbox.select(nd_offspring + nd_population, N_POP_ND)

        #stopping criteria
        if(time.time() - start_time > 420):
            print("Evolutionary run timed out")
            break
        
        if(top_nd_individual != nd_population[0]):
            top_nd_individual_age = 0
            top_nd_individual = nd_population[0]
        else:
            top_nd_individual_age += 1
            
        if(top_nd_individual_age == 15):
            print("Top ensemble has not changed in 15 generations - stopping the run")
            break
        
        generation_end_time = time.time()
        print(f"generation took {generation_end_time-generation_start_time}s")

        
    del creator.ND_Individual
    del creator.ND_Fitness
    return nd_population



def simple_ndea_experiment(task_id):
    BASE_LEARNER_CLASS = LogisticRegression
    ENSEMBLE_SIZE = 1
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

    
    accuracies = []
    start = time.time()
    for i in range(1 ):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=i, sample=0)
        X_train, y_train, X_test, y_test = X[train_indices], y[train_indices], X[test_indices], y[test_indices]
        ensemble = Ensemble()
        for j in range(ENSEMBLE_SIZE):
            top_individual_blueprint = simple_ndea(X_train, y_train, BASE_LEARNER_CLASS)[0].tree
            new_nd = NestedDichotomie(BASE_LEARNER_CLASS)
            new_nd = new_nd.fit(X_train, y_train, top_individual_blueprint)
            ensemble.append(new_nd)
            print(f"Created {j}th ND for {i}th ensemble")
        predictions = ensemble.predict(X_test)
        print(predictions)
        print(y_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
    print(f"Achieved average accuracy: {sum(accuracies)/len(accuracies)} in {time.time() - start}s")
             
        


#simple_ndea_experiment(3022)

import smtplib, ssl
def log_result(method, task, accuracy, duration, final_ensemble):
    try:
        final_ens_str = str([str(nd) for nd in final_ensemble])
    except:
        final_ens_str = "Representation failed"
        
    log_message = f"Experiment: {method} on task {task} \naccuracy: {accuracy}, in {duration} seconds, final ensemble:{final_ens_str}\n"
    with open("log.txt", "a") as f:
        f.write(log_message)
    


#single run times
start_time = time.time()
task_id = 3022
task = openml.tasks.get_task(task_id)
dataset = openml.datasets.get_dataset(task.dataset_id)
X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")

#preprocessing
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

train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
X_train, y_train, X_test, y_test = X[train_indices], y[train_indices], X[test_indices], y[test_indices]
top_individual_blueprint = simple_ndea(X_train, y_train, LogisticRegression)[0].tree
new_nd = NestedDichotomie(LogisticRegression)
new_nd = new_nd.fit(X_train, y_train, top_individual_blueprint)
prediction = new_nd.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
end_time = time.time()
log_result("test", task_id, accuracy, end_time - start_time, [new_nd])



