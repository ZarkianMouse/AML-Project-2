##############
# logstic.py #
##############

import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

def _logistic(X, y, lookup_fall_type):
    print("\n__logistic regression__")
    # random_state: set seed for random# generator
    # test_size: default 25% testing, 75% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

    lr = LogisticRegression(random_state=40)
    lr.fit(X_train, y_train)

    #########################################
    # Coefficients of linear model (b_1,b_2,...,b_p): log(p/(1-p)) = b0+b_1x_1+b_2x_2+...+b_px_p
    print("lr.coef_: {}".format(lr.coef_))
    print("lr.intercept_: {}".format(lr.intercept_))

    # Estimate the accuracy of the classifier on future data, using the test data
    ##########################################################################################
    print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

    ## Use the trained logistic regression classifier model to classify new, previously unseen falls
    # first example: a small fruit with mass 20g, color_score = 5.5, width 4.3 cm, height 5.5 cm
    fall_prediction = lr.predict([[4722.92, 4019, -1.61E+03, 14, 78, 319]])
    print(lookup_fall_type[fall_prediction[0]])

    # Estimate the accuracy of the classifier on future data, using the test data
    lr.score(X_test, y_test)
    lr_predict = lr.predict(X_test)
    print("The prediction", lr_predict)
    print("Logistic Regression Confusion Matrix:\n")
    print(confusion_matrix(y_test, lr_predict))