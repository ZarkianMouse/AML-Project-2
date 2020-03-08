"""
Applied Machine Learning Project_2: Using Logistic regression and Support Vector Machines (SVM)

COPYRIGHT (C) 2020 John Fahringer (jrf5001@yahoo.com) and Naomi Burhoe (naomi.burhoe@hotmail.com)
All rights reserved.

This program tries to classify a portion of our selected data set using the mentioned supervised
learning algorithms.
"""
import pandas as pd

from knn import _knn
from logistic import _logistic
from SVM import _SVM

falldetection = pd.read_csv('data/falldetection.csv')
falldetection.head()

# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fall_type = dict(zip(falldetection.ACTIVITY.unique(), falldetection.NAME.unique()))
print(lookup_fall_type)

X = falldetection[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']]
y = falldetection['ACTIVITY']

_knn(X, y)
_logistic(X, y)
_SVM(X, y)

