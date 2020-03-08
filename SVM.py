##############
#   SVM.py   #
##############

import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC
from sklearn.svm import SVC

def _SVM(X, y):
    print("__SVM__")