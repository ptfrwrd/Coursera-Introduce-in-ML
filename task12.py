import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import sklearn.model_selection as mod_sel


data = pd.read_csv("abalone.csv")
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
target, signs = data['Rings'], data[['Sex','Length','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']]
results = []


for i in range(1, 51):
    clf = RandomForestRegressor(n_estimators=i, random_state=1)
    sample_generator = KFold(n_splits=5, random_state=1, shuffle=True)
    score = np.mean(cross_val_score(X=signs, y=target, estimator=clf, cv=sample_generator, scoring='r2'))
    print(i, "  ", score)

