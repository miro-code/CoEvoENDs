import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class BinaryTreeNode:
    def __init__(self, value = None, left = None, right = None):
        self.value = value
        self.left = left
        self.right = right
        
    def __str__(self, level = 0):
        result = level * "\t" + str(self.value) + "\n"
        if(self.left is not None):
            result += self.left.__str__(level + 1)
            
        if(self.right is not None):
            result += self.right.__str__(level + 1)
        return result 
    def __eq__(self, other):
        if isinstance(other, BinaryTreeNode):
            return self.value == other.value and self.left.__eq__(other.left) and self.right.__eq__(other.right)
        return False

def i_max_random_tiebreak(array):
    v_max = array[0]
    i_max = [0]
    for i in range(1, len(array)):
        if(array[i] > v_max):
            v_max = array[i]
            i_max = [i]
        elif(array[i] == v_max):
            i_max.append(i)
    return random.choice(i_max)
    

def max_bincount_random_tiebreak(array):
    bc = np.bincount(array)
    return i_max_random_tiebreak(bc)


class Ensemble(list):

    def predict_proba(self, X):
        proba = self[0].predict_proba(X)
        for i in range(1,len(self)):
            proba += self[i].predict_proba(X)
        return proba/len(self)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.apply_along_axis(i_max_random_tiebreak, 1, proba)
    
    def refit(self, X, y):
        for nd in self:
            nd.fit(X,y,nd.tree)
        
    
class DecisionStump(DecisionTreeClassifier):
    def __init__(self, *, criterion="gini", splitter="best", max_depth=1, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0, class_weight=None, ccp_alpha=0):
        return super().__init__(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease, class_weight, ccp_alpha)