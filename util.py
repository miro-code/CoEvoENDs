from multiprocessing.sharedctypes import Value
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
    def __init__(self):
        super().__init__(max_depth=1)


class ConstBaseLearner:

    def fit(self, X, y):
        if(y[0] != 0 and y[0] != 1):
            raise ValueError("ConstBaseLearner can only be fitted to 0 or 1")
        self.y = y[0]

    def predict_proba(self, X):
        result = [[0, 0]]
        result[0][self.y] += 1
        return np.array(result)
