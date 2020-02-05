from sklearn import metrics

import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


x_test = pandas.read_csv('perceptron-test.csv', header=None).loc[:, 1:]
y_test = pandas.read_csv('perceptron-test.csv', header=None).loc[:, 0]
x_train = pandas.read_csv('perceptron-train.csv', header=None).loc[:, 1:]
y_train = pandas.read_csv('perceptron-train.csv', header=None).loc[:, 0]

clf = Perceptron(random_state=241, max_iter=5, tol=None)
clf.fit(x_train,y_train)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(float))
x_test_scaled = scaler.transform(x_test.astype(float))

accuracy_1 = metrics.accuracy_score(y_test, clf.predict(x_test))

clf = Perceptron(random_state=241, max_iter=5, tol=None)
clf.fit(x_train_scaled,y_train)
accuracy_2 = metrics.accuracy_score(y_test, clf.predict(x_test_scaled))
print(accuracy_1, " ", accuracy_2, " ", accuracy_2-accuracy_1)
