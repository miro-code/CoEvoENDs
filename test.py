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
# =============================================================================


n1.fit(X_train, y_train)

print(n1.predict_proba_single(X_test.iloc[:1, :]))
print(n1.predict_proba(X_test))

x = n1.predict_proba(X_test)
x.apply(sum)
