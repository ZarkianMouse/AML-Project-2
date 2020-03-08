"""
Applied Machine Learning Project_2: Using Logistic regression and Support Vector Machines (SVM)

COPYRIGHT (C) 2020 John Fahringer (jrf5001@yahoo.com) and Naomi Burhoe (naomi.burhoe@hotmail.com)
All rights reserved.

This program tries to classify a portion of our selected data set using the mentioned supervised
learning algorithms.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from knn import _knn
from logistic import _logistic
from SVM import _SVM

falldetection = pd.read_csv('data/falldetection.csv')
falldetection.head()

# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fall_type = dict(zip(falldetection.ACTIVITY.unique(), falldetection.NAME.unique()))
print(lookup_fall_type)

X = falldetection[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']]
y = falldetection['NAME']

_knn(falldetection, X, y)
_logistic(falldetection, X, y)
_SVM(falldetection, X, y)

