# -*- coding: utf-8 -*-
import random, util, os, sys
import numpy as np
from deap import base, creator, tools

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from nested_dichotomies import NestedDichotomie
from genotype import DistanceMatrix
from util import BinaryTreeNode, Ensemble, DecisionStump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import time, openml

#unseriös
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def conda(X, y, base_learner_class):
    
    N_VALID_SPLITS = 5
    VALID_SIZE = 0.1
    
    N_POP_ND = 16
    N_POP_ENS = 20
    N_GEN = 200
    CX_PB_ND, MUT_PB_ND = 1, 1
    MUTATE_ETA_ND = 30
    CX_PB_ENS, MUT_PB_ENS = 0.7, 1
    
    MUT_APPEND_PB_ENS, MUT_REDUCE_PB_ENS, MUT_ALTER_PB_ENS = 0.5, 0.1, 0.9
    JOIN_ENS_TOURNAMENT_SIZE = 2
    OFFSPRING_TOURNAMENT_SIZE = 2
    N_ENS_ASSIGNING_SUPPORT = N_POP_ENS
    
    EXPECTED_INIT_ENSEMBLE_SIZE = 20
    N_TIMES_MORE_ENS_GENS = 5
    RESET_INTERVAL = 5 
    TIMEOUT = 420*10*2
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
        accuracy = sum(scores)/len(scores)
        return accuracy, len(individual)
    
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
        if(len(individual) > 1 and random.random() < MUT_REDUCE_PB_ENS):
            i = random.randint(0, len(individual) - 1)
            #todo: fitness dependent selection
            individual.pop(i)
        if(len(individual) < 40 and random.random() < MUT_APPEND_PB_ENS):
            new_nd = toolbox.join_ens_tournament(nd_population, 1)[0] #tournSel retourns list of selected individuals
            individual.insert(random.randint(0, len(individual)), new_nd)
        if(random.random() < MUT_ALTER_PB_ENS):
            i = random.randint(0, len(individual) - 1)
            #todo: fitness dependent selection
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

    top_ens_individual = None
    start_time = time.time()
    
    
    for g in range(N_GEN):
        
        generation_start_time = time.time()
        
        #reset
        if(g and g % RESET_INTERVAL == 0):
            nd_population[1:] = toolbox.nd_population(N_POP_ND-1)
            toolbox.evaluate_population(toolbox.evaluate_nd, nd_population[1:])
            
            n_elite = N_POP_ENS // 2
            ens_population[n_elite:] = toolbox.ens_population(N_POP_ENS - n_elite, nd_population)
            toolbox.evaluate_population(toolbox.evaluate_ens, ens_population[n_elite:])
            
        
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
        nd_population[:] = toolbox.select(nd_population+nd_offspring, N_POP_ND)

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

        if(time.time() - start_time > TIMEOUT):
            print("Evolutionary run timed out after {g+1} generations")
            generation_end_time = time.time()
            print(f"generation took {generation_end_time-generation_start_time}s")
            break
        
        if(top_ens_individual != ens_population[0]):
            top_ens_individual_age = 0
            top_ens_individual = nd_population[0]
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
    
    N_POP_ND = 16
    N_GEN = 1 #CHANGE
    CX_PB_ND, MUT_PB_ND = 1, 1
    MUTATE_ETA_ND = 30
    
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

        #stopping criteria
        if(time.time() - start_time > 420):
            print("Evolutionary run timed out after {g+1} generations")
            generation_end_time = time.time()
            print(f"generation took {generation_end_time-generation_start_time}s")
            break
        
        if(top_nd_individual != nd_population[0]):
            top_nd_individual_age = 0
            top_nd_individual = nd_population[0]
        else:
            top_nd_individual_age += 1
            
        if(top_nd_individual_age == 15):
            print("Top individual has not changed in 15 generations - stopping the run")
            break
        
        generation_end_time = time.time()

        
    del creator.ND_Individual
    del creator.ND_Fitness
    return nd_population


def log_result(experiment_id, method, base_learner, task, fold, accuracy, train_accuracy, ensemble_size, duration, other_results = None):
    log_message = "experiment id: {}, method: {}, base_learner: {} task: {}, fold: {}, accuracy: {}, train_accuracy: {}, ensemble_size: {}, duration: {}, other_results = {}\n".format(experiment_id, method, base_learner, task, fold, accuracy, train_accuracy, ensemble_size, duration, other_results)
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'results\\results.txt')
    
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
        ENSEMBLE_SIZE = 1 # CHANGE TO 10 DONT FORGET
        top_ensemble = Ensemble()
        for j in range(ENSEMBLE_SIZE):
            top_individual=simple_ndea(X_train, y_train, base_learner)[0]
            top_individual_blueprint = top_individual.tree
            train_accuracy=top_individual.fitness.values[0]
            new_nd = NestedDichotomie(base_learner)
            new_nd = new_nd.fit(X_train, y_train, top_individual_blueprint)
            top_ensemble.append(new_nd)
    elif(method == "conda"):
        final_population = conda(X_train, y_train, base_learner)[1]
        top_ens_individual = final_population[0]
        train_accuracy = top_ens_individual.fitness.values[0]
        top_ensemble = Ensemble([nd_genotype.phenotype for nd_genotype in top_ens_individual])
        top_ensemble.refit(X_train,y_train)

        for i in range(1, len(final_population)):
            ens_individual = final_population[i]
            train_accuracy = ens_individual.fitness.values[0]
            ensemble = Ensemble([nd_genotype.phenotype for nd_genotype in ens_individual])
            ensemble.refit(X_train,y_train)
            ensemble_predictions = ensemble.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
            other_results.append(("size: "+ str(len(ensemble)), "test accuracy:" +str(ensemble_accuracy), "train accuracy: " + str(train_accuracy)))
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
    base_learners = [DecisionTreeClassifier, DecisionStump]
    methods = ["ndea", "conda"]
    experiment_configurations = [(method, task_id, fold_id, base_learner) for method in methods for task_id in tasks for fold_id in range(10) for base_learner in base_learners]
    
    experiment_id = 360#int(sys.argv[1]) ÄNDERN 
    if(experiment_id < 0 or experiment_id > len(experiment_configurations)-1):
        raise ValueError("Illegal experiment ID")
    
    
    single_experiment(*experiment_configurations[experiment_id], experiment_id)
    


def test():
    tasks = [7, 9, 40, 146204, 18, 9964, 41, 3022, 145681, 2]
    base_learners = [DecisionTreeClassifier, DecisionStump]
    methods = ["ndea", "conda"]
    experiment_configurations = [(method, task_id, fold_id, base_learner) for method in methods for task_id in tasks for fold_id in range(10) for base_learner in base_learners]
    
    for experiment_id in range(len(experiment_configurations)):
        print(experiment_configurations[experiment_id])
        single_experiment(*experiment_configurations[experiment_id], experiment_id)
    

main()


#print(len(experiment_configurations))

# =============================================================================
# 
# 
# #single run times
# start_time = time.time()
# task_id = 6
# task = openml.tasks.get_task(task_id)
# dataset = openml.datasets.get_dataset(task.dataset_id)
# X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
# #preprocessing
# categorical_features = list(X.columns[categorical_indicator])
# numeric_features = list(X.columns[~np.array(categorical_indicator)])
# numeric_transformer = SimpleImputer(strategy="median")
# categorical_transformer = OneHotEncoder(sparse = False)
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, numeric_features),
#         ("cat", categorical_transformer, categorical_features),
#     ]
# )
# X = preprocessor.fit_transform(X)
# 
# train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
# X_train, y_train, X_test, y_test = X[train_indices], y[train_indices], X[test_indices], y[test_indices]
# top_individual_blueprint = simple_ndea(X_train, y_train, LogisticRegression)[0].tree
# new_nd = NestedDichotomie(LogisticRegression)
# new_nd = new_nd.fit(X_train, y_train, top_individual_blueprint)
# prediction = new_nd.predict(X_test)
# accuracy = accuracy_score(y_test, prediction)
# end_time = time.time()
# log_result("test", task_id, accuracy, end_time - start_time, [new_nd])
# 
# 
# 
# from sklearn.tree import DecisionTreeClassifier
# 
# 
# dt = DecisionTreeClassifier()
# start = time.time()
# dt.fit(X,y)
# end = time.time()
# print(end-start)
# start = time.time()
# dt.fit(X,y)
# end = time.time()
# print(end-start)
# print(X.shape)
# 
# new_nd = NestedDichotomie(LogisticRegression)
# fitstart = time.time()
# new_nd = new_nd.fit(X_train, y_train, top_individual_blueprint)
# fitend = time.time()
# prediction = new_nd.predict(X_test)
# predend = time.time()
# accuracy = accuracy_score(y_test, prediction)
# print(f"fitting took {fitend - fitstart}s and prediction took {predend - fitend}s")
# 
# 
# 
# 
# 
# =============================================================================
