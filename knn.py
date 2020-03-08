##############
#   knn.py   #
##############

import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def _knn(X, y):
    print ("__KNN__")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.75, random_state=0)

    # plotting a scatter matrix
    cmap = cm.get_cmap('gnuplot')
    scatter = scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)

    # Create classifier object
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

    # Train the classifier (fit the estimator) using the training data
    knn.fit(X_train, y_train)

    # Estimate the accuracy of the classifier on future data, using the test data
    knn.score(X_test, y_test)
    knn_predict = knn.predict(X_test)
    print("The prediction", knn_predict)
    print("KNN Confusion Matrix:")
    print(confusion_matrix(y_test, knn_predict))

    # How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?
    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0, 5, 10, 15, 20])
    plt.show()