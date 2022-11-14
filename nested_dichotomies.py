# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import random, util
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class NestedDichotomie:
    def  __init__(self, base_learner_class):
        self.base_learner_class = base_learner_class
        
    def __str__(self, level = 0):
        result = level * "\t" + str(self.classes_) + "\n"
        if(self.left is not None):
            result += self.left.__str__(level + 1)
            
        if(self.right is not None):
            result += self.right.__str__(level + 1)
        return result 
    
    
    def fit(self, X, y, tree):
        self.tree = tree
        if(len(tree.value) == 1):
            return
        self.zero_dichotomy = False
        if(len(y) == 0):
            self.zero_dichotomy = True
            return
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y) #dont use (only for sklearn) - if classes are missing from y this could lead to problems
        self.classes = unique_labels(list(tree.value))
        self.left = None
        self.right = None
        if(self.tree.left is not None):
            self.left = NestedDichotomie(self.base_learner_class)
            self.right = NestedDichotomie(self.base_learner_class)
        X_train_left = []
        X_train_right = []
        y_left = []
        y_right = []
        y_train = []
        for i in range(len(y)):
            if(y[i] in self.tree.left.value):
                X_train_left.append(X[i, :])
                y_train.append(0)
                y_left.append(y[i])
            elif(y[i] in self.tree.right.value):
                X_train_right.append(X[i, :])
                y_train.append(1)
                y_right.append(y[i])
        

        if(len(y_right) > 0 and len(y_left) > 0):
            self.base_learner = self.base_learner_class()
        else:
            self.base_learner = util.ConstBaseLearner() #in case only data points of a single class are available

        self.base_learner.fit(np.array(X), y_train)
        if(self.left is not None):
            self.left.fit(X_train_left, y_left, self.tree.left)
            self.right.fit(X_train_right, y_right, self.tree.right)
        return self
    
    def predict_proba_helper_(self, x):
        if(len(self.tree.value) == 1):
            return {list(self.tree.value)[0] : 1}
        if(self.zero_dichotomy):
            return {cls : 0 for cls in list(self.tree.value)}
        x = x.reshape(1, -1)
        left_dict = self.left.predict_proba_helper_(x)
        right_dict = self.right.predict_proba_helper_(x)
        left_proba, right_proba = self.base_learner.predict_proba(x)[0]
        for key in left_dict:
            left_dict[key] *= left_proba
        for key in right_dict:
            right_dict[key] *= right_proba 
        return left_dict | right_dict
    
    def predict_proba_single(self, x):
        dic = self.predict_proba_helper_(x)
        classes = list(dic)
        classes.sort()
        result = [dic[key] for key in classes]
        return result
        
    def predict_proba(self, X):
        return np.apply_along_axis(self.predict_proba_single, 1, X)
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        probas = self.predict_proba(X)
        return np.apply_along_axis(self.get_probable_class, 1, probas)
        
    def get_probable_class(self, probas):
        i = util.i_max_random_tiebreak(probas)
        return self.classes[i]
        
    
    def get_estimate(self, x, class_name):
        if(class_name not in self.tree.value):
            print("class not represented")
            return 0
        if(len(self.tree.value) == 1):
            return 1
        if(class_name in self.left.tree.value):
            return self.base_learner.predict_proba(x)[0][0] * self.left.get_estimate(x, class_name) #prediction for first datapoint 
        
        elif(class_name in self.right.tree.value):
            return self.base_learner.predict_proba(x)[0][1] * self.left.get_estimate(x, class_name)
        else:
            print("Error in tree structure")
            return 0
        
    def get_params(self, deep = True):
        return {
            "base_learner_class" : self.base_learner_class,
            "tree" : self.tree
            }
        
