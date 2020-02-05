import pandas
import numpy as np
from sklearn.metrics import roc_auc_score


def euclidean(w1, w2, w1_end, w2_end):
   return (((w1 - w1_end)**2 + (w2 - w2_end)**2) ** 0.5)


data = pandas.read_csv('data-logistic.csv', header=None)
x1_data = data.iloc[:, 1]
x2_data = data.iloc[:, 2]
y_data = data.iloc[:, 0]
k = 0.1
w1, w2, w1_prev, w2_prev = 0, 0, 0, 0
C = 10
for j in range(10000):
        w1 += k*np.mean(y_data[:] * x1_data[:] * (1 - 1 / (1 + np.exp(- y_data[:] * (w1 * x1_data[:] + w2 * x2_data[:]))))) - k * C * w1
        w2 += k*np.mean( y_data[:] * x2_data[:] * (1 - 1 / (1 + np.exp(- y_data[:] * (w1 * x1_data[:] + w2 * x2_data[:]))))) - k * C * w2
        if euclidean(w1, w2, w1_prev, w2_prev) < 1e-5:
            break
        w1_prev = w1
        w2_prev = w2


w = np.array([w1, w2])
X = np.array([x1_data, x2_data])
sigmoid = 1 / (1 + np.exp(-w.dot(X)))
print(roc_auc_score(y_data, sigmoid), '\n')

