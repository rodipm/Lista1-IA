# """
# Questão 2: KNN com 10 vizinhos
# Validação cruzada KFold (sem estratificação) com K = 5
# """

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('class02.csv')
X = dataset.iloc[:, 0:100].values
y = dataset.iloc[:, -1].values

# KNN classifier - 10 neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=10, metric='euclidean')

# Split dataset into 5 folds
k_fold = KFold(n_splits=5, shuffle=False)

# KFold repetitions
scores = []
for train_index, test_index in k_fold.split(X):
    # get indexes
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # train
    knn_classifier.fit(X_train, y_train)
    
    # predict
    scores.append(knn_classifier.score(X_test, y_test))


print("==== KNN =====")
print("Valores médios por repetição: ")
print(*["\t" + str(x+1) + ".: " + str(scores[x]) + "\n" for x in range(5)])
print("Valor médio da acurácia: ", np.mean(scores))