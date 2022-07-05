# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:51:33 2022

@author: pro
"""

from util import BinaryTreeNode

class Distance:
    def __init__(self, c1, c2, value):
        self.class1 = c1
        self.class2 = c2
        self.value = value
        
    def get_classes(self):
        return [self.class1, self.class2]
    
    def __str__(self):
        return str(self.class1) + " - " + str(self.class2) + ": " + str(self.value)

class DistanceMatrix:
    
    def __init__(self, classes = None, distances_raw = None):
        self.classes = classes
        self.class_to_index = {k: v for v, k in enumerate(classes)}
        self.distances = self.encode_distances_(distances_raw)
        self.distances.sort(key = lambda distance:distance.value)
        
    def encode_distances_(self, distances_raw):
        pointer = 0
        result = []
        for i in range(len(self.classes)-1):
            for j in range(1+i, len(self.classes)):
                result.append(Distance(self.classes[i], self.classes[j], distances_raw[pointer]))
                pointer += 1
        return result

    
    def build_tree(self):
        nodes = {c:BinaryTreeNode({c}) for c in self.classes}
        distinct_nodes = len(self.classes)
        
        for dist in self.distances:
            c1 = dist.class1
            c2 = dist.class2
            if(nodes[c1] == nodes[c2]):
                continue
            new_node = BinaryTreeNode(nodes[c1].value | nodes[c2].value, nodes[c1], nodes[c2])
            for c in nodes[c1].value:
                nodes[c] = new_node
            for c in nodes[c2].value:
                nodes[c] = new_node
            distinct_nodes -= 1
            if(distinct_nodes < 2):
                break
        return nodes.popitem()[1] #pops key value pair
        

