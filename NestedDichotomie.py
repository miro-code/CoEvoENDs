# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:11:35 2022

@author: pro
"""
import pandas as pd 

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