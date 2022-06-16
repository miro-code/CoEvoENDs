import random
import numpy as np

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
    