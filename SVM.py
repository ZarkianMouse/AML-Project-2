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
    print("\n__SVM__")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

    # partition the data into two classes
    y_train_1 = y_train == 1  # Walking in True class, others in False class
    y_test_1 = y_test == 1  # Walking in True class, others in False class
    y_train = 2 - y_train_1  # Walking = 1; others =2
    y_test = 2 - y_test_1

    seeData = True
    if seeData:
        # plotting a scatter matrix
        cmap = cm.get_cmap('gnuplot')
        scatter = scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9),
                                 cmap=cmap)

        # plotting a 3D scatter plot
        from mpl_toolkits.mplot3d import axes3d  # must keep
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_train['SL'], X_train['BP'], X_train['HR'], c=y_train, marker='o', s=100)
        ax.set_xlabel('Sugar Level')
        ax.set_ylabel('Blood Pressure')
        ax.set_zlabel('Heart beat rate')
        plt.show()

    # Create classifier object: Create a linear SVM classifier
    # C: Regularization parameter. Default C=1
    lsvc = LinearSVC(C=100, random_state=10, tol=1e-4)
    lsvc.fit(X_train, y_train)
    print("Linear SVM Training set score: {:.2f}%".format(100 * lsvc.score(X_train, y_train)))
    print("Linear SVM Test set score: {:.2f}%".format(100 * lsvc.score(X_test, y_test)))

    #
    lsvc.predict(X_test)
    print(lsvc.coef_)
    print(lsvc.intercept_)

    # Estimate the accuracy of the classifier on future data, using the test data
    lsvc_predict = lsvc.predict(X_test)
    print("\nLinear SVM Confusion Matrix:")
    print(confusion_matrix(y_test, lsvc_predict), '\n')

    # Create classifier object: Create a nonlinear SVM classifier
    # kernel, default=’rbf’ = radial basis function
    # if poly, default degree = 3
    ### NOTE: This took my computer over an hour and a half to run.
    _exec = True
    if _exec:
        svc = SVC(degree=2, kernel='poly', random_state=1)
        svc.fit(X_train, y_train)
        print("SVM Poly Training set score: {:.2f}%".format(100 * svc.score(X_train, y_train)))
        print("SVM Poly Test set score: {:.2f}%".format(100 * svc.score(X_test, y_test)))

        svc_predict = svc.predict(X_test)
        print("\nNONLinear SVM Confusion Matrix:")
        svm_cm = confusion_matrix(y_test, svc_predict)
        print(svm_cm, '\n')

        # plotting the confusion matrix
        import seaborn as sns
        fig, ax = plt.subplots()
        cbar_ax = fig.add_axes([.92, .3, .02, .4])
        sns.heatmap(svm_cm, ax=ax, cmap="Greens", annot=True, cbar_ax=cbar_ax)
        ax.set_title('Confusion Matrix NONLinear SVM Classifier')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.xaxis.set_ticklabels(['Fall', 'Not Fall'])
        ax.yaxis.set_ticklabels(['Fall', 'Not Fall'])

    # Create classifier object: Create a nonlinear SVM classifier
    # kernel, default=’rbf’ = radial basis function
    svc = SVC(C=10)
    svc.fit(X_train, y_train)
    print("SVM Gaussian Training set score: {:.2f}%".format(100 * svc.score(X_train, y_train)))
    print("SVM Gaussian Test set score: {:.2f}%".format(100 * svc.score(X_test, y_test)))

    svc_predict = svc.predict(X_test)
    print("\nNONLinear SVM Confusion Matrix:")
    print(confusion_matrix(y_test, svc_predict), '\n')

    # SVM for multiple classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

    # SVM with linear kernel
    ### NOTE: This took my computer took a while to run.
    _exec = True
    if _exec:
        svc = SVC(C=10, degree=1, kernel='poly')
        svc.fit(X_train, y_train)
        print("SVM Gaussian Training set score: {:.2f}%".format(100 * svc.score(X_train, y_train)))
        print("SVM Gaussian Test set score: {:.2f}%".format(100 * svc.score(X_test, y_test)))

        svc_predict = svc.predict(X_test)
        print("\nLinear SVM Confusion Matrix:")
        svm_cm = confusion_matrix(y_test, svc_predict)
        print(svm_cm, '\n')

        # plotting the confusion matrix
        import seaborn as sns
        fig, ax = plt.subplots()
        cbar_ax = fig.add_axes([.92, .3, .02, .4])
        sns.heatmap(svm_cm, ax=ax,cmap="Greens", annot=True, cbar_ax=cbar_ax)
        ax.set_title('Confusion Matrix Linear SVM Classifier')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.xaxis.set_ticklabels(['Fall', 'Sit', 'Cramps', 'Run', 'Stand', 'Walk'])
        ax.yaxis.set_ticklabels(['Fall', 'Sit', 'Cramps', 'Run', 'Stand', 'Walk'])
