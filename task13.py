import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


data = pd.read_csv("gbm-data.csv")
y_data = data["Activity"].values
x_data = data.iloc[:, 1:].values
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.8, random_state=241)
learning_rates = [1, 0.5, 0.3, 0.2, 0.1]
learning_rates = [0.2]
loss_test, loss_train = [],[]

gb = GradientBoostingClassifier(n_estimators=119, verbose=True, random_state=241)
gb.fit(x_train, y_train)
pred = gb.predict_proba(x_test)
print(log_loss(y_test, pred))
