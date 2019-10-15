""" 
Questão 3: Lasso Regression
"""

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

# Import dataset
dataset = pd.read_csv('reg01.csv')
X = dataset.iloc[:, 0:10].values
y = dataset.iloc[:, -1].values

# Split dataset into train and test sets using kfold approach
loo = LeaveOneOut()

rmse_train_history = []
rmse_test_history = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Gaussian Naive Bayes Classifier
    regressor = linear_model.Lasso(alpha=1.0)
    regressor.fit(X_train, y_train)

    # Prediction Test Sets
    y_train_pred = regressor.predict(X_train)
    y_pred = regressor.predict(X_test)

    # Prediction Visualization
    rmse_train = sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = sqrt(mean_squared_error(y_test, y_pred))

    rmse_train_history.append(rmse_train)
    rmse_test_history.append(rmse_test)


rmse_train_mean = np.mean(rmse_train_history)
rmse_test_mean = np.mean(rmse_test_history)

print("===== LASSO REGRESSION =====")
print("Mean RMSE:")
print("Train Set: ", rmse_train_mean)
print("Test Set: ", rmse_test_mean)
