import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas

data = pandas . read_csv('train.csv', index_col='PassengerId')
data['Sex'].value_counts()
print("577 314")
data['Survived'].value_counts()
print(round((342/(549+342))*100, 2))
data['Pclass'].value_counts()
print(round(216/(216+491+184)*100, 2))
print(data['Age'].mean(), data['Age'].median())
print(data['SibSp'].corr(data['Parch']))


