# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:11:35 2022

@author: pro
"""
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
        
        
#TEST 
# =============================================================================
#         
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# 
# from util import *
# 
# t1 = BinaryTreeNode([0, 1, 2])
# t1.left = BinaryTreeNode([0, 1])
# t1.left.left = BinaryTreeNode([0])
# t1.left.right = BinaryTreeNode([1])
# t1.right = BinaryTreeNode([2])
# 
# 
# t2 = BinaryTreeNode([0, 1, 2])
# t2.left = BinaryTreeNode([0, 2])
# t2.left.left = BinaryTreeNode([0])
# t2.left.right = BinaryTreeNode([2])
# t2.right = BinaryTreeNode([1])
# 
# t3 = BinaryTreeNode([0, 1, 2])
# t3.left = BinaryTreeNode([1, 2])
# t3.left.left = BinaryTreeNode([1])
# t3.left.right = BinaryTreeNode([2])
# t3.right = BinaryTreeNode([0])
# 
# t11 = BinaryTreeNode([0, 1, 2])
# t11.left = BinaryTreeNode([0, 1])
# t11.left.left = BinaryTreeNode([0])
# t11.left.right = BinaryTreeNode([1])
# t11.right = BinaryTreeNode([2])
# 
# n1 = NestedDichotomie(LogisticRegression)
# n2 = NestedDichotomie(LogisticRegression)
# 
# 
# 
# df = pd.read_csv("C:\\Users\\pro\\Downloads\\iris.data")
# 
# outcomes = df.iloc[:,4]
# 
# le = LabelEncoder()
# outcomes = le.fit_transform(outcomes)
# 
# features = df.drop(df.columns[[4]], axis = 1)
# 
# #achtung beim erstellen von dummy variablen bei klassen mit vielen ausprägungen
# #features = pd.get_dummies(features)
# 
# #sinnvollen wert zum füllen von nans finden
# #features = features.fillna(0.0)
# 
# 
# X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size = 0.3) 
# 
# n1.fit(X_train, y_train, t1)
# n2.fit(X_train, y_train, t2)
# 
# nd_train_pred = n1.predict(X_train)
# nd_test_pred = n1.predict(X_test)
# nd_train_accuracy = accuracy_score(y_train, nd_train_pred)
# nd_test_accuracy = accuracy_score(y_test, nd_test_pred)
# print('The dt training accuracy is', nd_train_accuracy)
# print('The dt test accuracy is', nd_test_accuracy)
# 
# =============================================================================

# =============================================================================
# 
# lr_model = LogisticRegression().fit(X_train, y_train)
# 
# 
# lr_train_pred = lr_model.predict(X_train)
# lr_test_pred = lr_model.predict(X_test)
# 
# lr_train_accuracy = accuracy_score(y_train, lr_train_pred)
# lr_test_accuracy = accuracy_score(y_test, lr_test_pred)
# print('The logistic regression training accuracy is', lr_train_accuracy)
# print('The logistic regression test accuracy is', lr_test_accuracy)
# 
# 
# dt_train_pred = n1.predict(X_train)
# dt_test_pred = n1.predict(X_test)
# 
# dt_train_accuracy = accuracy_score(y_train, dt_train_pred)
# dt_test_accuracy = accuracy_score(y_test, dt_test_pred)
# print('The dt training accuracy is', dt_train_accuracy)
# print('The dt test accuracy is', dt_test_accuracy)
# 
# l1 = le.inverse_transform(dt_test_pred)
# l2 = le.inverse_transform(lr_test_pred)
# 
# =============================================================================
