# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:11:35 2022

@author: pro
"""
import pandas as pd 
import random

class NestedDichotomie:
    def  __init__(self, learner_class, node):
        self.classes = node.value
        self.classes.sort()
        self.base_learner = learner_class()
        
        self.left = None
        self.right = None
        if(node.left is not None):
            self.left = NestedDichotomie(learner_class, node.left)
            self.right = NestedDichotomie(learner_class, node.right)
        self.X_train = pd.DataFrame()
        self.y_train = []  
        
    def __str__(self, level = 0):
        result = level * "\t" + str(self.classes) + "\n"
        if(self.left is not None):
            result += self.left.__str__(level + 1)
            
        if(self.right is not None):
            result += self.right.__str__(level + 1)
        return result 
    
    
    def fit(self, X, y):
        if(len(self.classes) == 1):
            return
        for i in range(len(y)):
            if(y[i] in self.left.classes):
                self.X_train = self.X_train.append(X.iloc[i, :])
                self.y_train.append(0)
            elif(y[i] in self.right.classes):
                self.X_train = self.X_train.append(X.iloc[i, :])
                self.y_train.append(1)
        self.base_learner.fit(self.X_train, self.y_train)
        if(self.left is not None):
            self.left.fit(X,y)
            self.right.fit(X,y)
            
    def predict_proba_helper_(self, x):
        if(len(self.classes) == 1):
            return {self.classes[0] : 1}
        left_dict = self.left.predict_proba_helper_(x)
        right_dict = self.right.predict_proba_helper_(x)
        left_proba, right_proba = self.base_learner.predict_proba(x)[0]
        for key in left_dict:
            left_dict[key] *= left_proba
        for key in right_dict:
            right_dict[key] *= right_proba 
        
        return left_dict | right_dict
    
    def predict_proba_single(self, x):
        if(isinstance(x, pd.Series)):
            x = x.to_numpy().reshape(1,-1)
        dict = self.predict_proba_helper_(x)
        classes = list(dict)
        classes.sort()
        result = [dict[key] for key in classes]
        return result
        
    def predict_proba(self, X):
        return X.apply(self.predict_proba_single, axis = 1)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.apply(self.get_probable_class)
        
    def get_probable_class(self, probas):
        p_max = 0
        i_max = []
        for i in range(len(probas)):
            if(probas[i] > p_max):
                p_max = probas[i]
                i_max = [i]
            if(probas[i] == p_max):
                i_max.append(i)
        return self.classes[random.choice(i_max)]
        
    
    def get_estimate(self, x, class_name):
        if(class_name not in self.classes):
            print("class not represented")
            return 0
        if(len(self.classes) == 1):
            return 1
        if(class_name in self.left.clases):
            return self.base_learner.predict_proba(x)[0][0] * self.left.get_estimate(x, class_name) #prediction for first datapoint 
        
        elif(class_name in self.right.classes):
            return self.base_learner.predict_proba(x)[0][1] * self.left.get_estimate(x, class_name)
        else:
            print("Error in tree structure")
            return 0
        
        
        
#TEST 
        
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from NestedDichotomie import *
from BinaryTreeNode import *

t1 = BinaryTreeNode([0, 1, 2])
t1.left = BinaryTreeNode([0, 1])
t1.left.left = BinaryTreeNode([0])
t1.left.right = BinaryTreeNode([1])
t1.right = BinaryTreeNode([2])


t2 = BinaryTreeNode([0, 1, 2])
t2.left = BinaryTreeNode([0, 2])
t2.left.left = BinaryTreeNode([0])
t2.left.right = BinaryTreeNode([2])
t2.right = BinaryTreeNode([1])

t3 = BinaryTreeNode([0, 1, 2])
t3.left = BinaryTreeNode([1, 2])
t3.left.left = BinaryTreeNode([1])
t3.left.right = BinaryTreeNode([2])
t3.right = BinaryTreeNode([0])

n1 = NestedDichotomie(LogisticRegression, t1)



df = pd.read_csv("C:\\Users\\pro\\Downloads\\iris.data")

outcomes = df.iloc[:,4]

le = LabelEncoder()
outcomes = le.fit_transform(outcomes)

features = df.drop(df.columns[[4]], axis = 1)

#achtung beim erstellen von dummy variablen bei klassen mit vielen ausprägungen
#features = pd.get_dummies(features)

#sinnvollen wert zum füllen von nans finden
#features = features.fillna(0.0)


X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size = 0.3) 

lr_model = LogisticRegression().fit(X_train, y_train)


lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

lr_train_accuracy = accuracy_score(y_train, lr_train_pred)
lr_test_accuracy = accuracy_score(y_test, lr_test_pred)
print('The logistic regression training accuracy is', lr_train_accuracy)
print('The logistic regression test accuracy is', lr_test_accuracy)



n1.fit(X_train, y_train)


dt_train_pred = n1.predict(X_train)
dt_test_pred = n1.predict(X_test)

dt_train_accuracy = accuracy_score(y_train, dt_train_pred)
dt_test_accuracy = accuracy_score(y_test, dt_test_pred)
print('The dt training accuracy is', dt_train_accuracy)
print('The dt test accuracy is', dt_test_accuracy)

l1 = le.inverse_transform(dt_test_pred)
l2 = le.inverse_transform(lr_test_pred)