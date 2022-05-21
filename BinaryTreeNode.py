class BinaryTreeNode:
    def __init__(self, value = None):
        self.value = value
        self.left = None
        self.right = None
        
    def __str__(self, level = 0):
        result = level * "\t" + str(self.value) + "\n"
        if(self.left is not None):
            result += self.left.__str__(level + 1)
            
        if(self.right is not None):
            result += self.right.__str__(level + 1)
        return result 

    