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




#test single task in console
"""
import openml
task_id = 7
task = openml.tasks.get_task(task_id)
dataset = openml.datasets.get_dataset(task.dataset_id)
X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")



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

import tests

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
fold_id = 9
train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold_id, sample=0)
X_train, y_train, X_test, y_test = X[train_indices], y[train_indices], X[test_indices], y[test_indices]
fold_id = 8
train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold_id, sample=0)
X_train8, y_train8, X_test8, y_test8 = X[train_indices], y[train_indices], X[test_indices], y[test_indices]

tree9 = tests.random_tree(unique_labels(y_train))
nd = NestedDichotomie(DecisionTreeClassifier)
nd.fit(X_train, y_train, tree9)
pred = nd.predict(X_test)
accuracy = accuracy_score(y_test, pred)

ens = Ensemble([nd])


"""