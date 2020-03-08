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

def _logistic(X, y):
    print("__logistic regression__")