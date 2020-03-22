import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


data = pd.read_csv("close_prices.csv")
data = data.iloc[:, 1:]

jonse = pd.read_csv("djia_index.csv")
jonse = jonse.iloc[:, 1:]

x_for_PCA= np.array(data)
pca = PCA(n_components=10)
principalComponents = pca.fit(x_for_PCA)

count = 0
sum = 0
for i in range(len(pca.explained_variance_ratio_)):
    count += 1
    value = pca.explained_variance_ratio_[i]
    sum += value
    if sum > 0.9:
        break
print("Need components %d" % count)
first_comp = pd.DataFrame(pca.transform(x_for_PCA)[:, 0])
coef = np.corrcoef(first_comp.T, jonse.T)[1, 0]
print("Corrcoef %0.2f" % coef)
print( data.columns[ np.argmax( pca.components_[0] ) ] )