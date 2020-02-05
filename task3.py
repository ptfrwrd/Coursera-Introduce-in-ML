import pandas
import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

data = np.genfromtxt('wine.data', delimiter=',')
y = data[:,0]
x = data[:,1:]
# для задания 3-4 (масштабирование)
x = scale(x)

kf = KFold(n_splits = 5, shuffle = True, random_state=42)

for k in range(1,51):
    classifier = KNeighborsClassifier(n_neighbors=k)
    quality = cross_val_score(classifier, x, y, cv = kf, scoring='accuracy')
    print(k, " ", (quality[0]+quality[1]+quality[2]+quality[3]+quality[4])/5)


