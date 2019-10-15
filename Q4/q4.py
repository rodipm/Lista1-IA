""" 
Questão 3: Decision Tree Regression
"""

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

# Import dataset
dataset = pd.read_csv('reg02.csv')
X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, -1].values

# Split dataset into train and test sets using KFold approach
k_fold = KFold(n_splits=5)

mae_train_history = []
mae_test_history = []
for train_index, test_index in k_fold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Gaussian Naive Bayes Classifier
    regressor = DecisionTreeRegressor(criterion="mse")
    regressor.fit(X_train, y_train)

    # Prediction Test Sets
    y_train_pred = regressor.predict(X_train)
    y_pred = regressor.predict(X_test)

    # Prediction Visualization
    mae_train = sqrt(mean_absolute_error(y_train, y_train_pred))
    mae_test = sqrt(mean_absolute_error(y_test, y_pred))

    mae_train_history.append(mae_train)
    mae_test_history.append(mae_test)


mae_train_mean = np.mean(mae_train_history)
mae_test_mean = np.mean(mae_test_history)

print("===== Árvore de Regressão =====")
print("Valores MAE para base de treino:")
print(*["\t" + str(x+1) + ".: " + str(mae_train_history[x]) + "\n" for x in range(5)])
print("Valores MAE para base de testes:")
print(*["\t" + str(x+1) + ".: " + str(mae_test_history[x]) + "\n" for x in range(5)])
print("Valor médio do MAE:")
print("Train Set: ", mae_train_mean)
print("Test Set: ", mae_test_mean)
