
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston

boston = load_boston()
x = boston.data
y = boston.target
x = scale(x)
i = np.linspace(1.0, 10.0, num=200)
regression = KNeighborsRegressor(n_neighbors=5, weights='distance')
kf = KFold(n_splits = 5, shuffle=False, random_state=42)
classifier = KNeighborsRegressor(metric='minkowski',n_neighbors=5, weights='distance')
for p in i:
    classifier.p = p
    quality = cross_val_score(classifier, x, y, cv=kf, scoring='neg_mean_squared_error')
    print(p, " ", (quality[0] + quality[1] + quality[2] + quality[3] + quality[4]) / 5)