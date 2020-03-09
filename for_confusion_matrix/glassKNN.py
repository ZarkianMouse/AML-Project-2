#!/usr/bin/env python
# coding: utf-8

# # KNN Individual Report: The Glass Dataset
# By: Naomi Burhoe

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import pandas as pd
pd.set_option('display.width', 1000)
glass = pd.read_csv('data/glass.csv')
glass.head()


# ## Background
# The dataset which will be analyzed today revolves around types of glass. Apparently, this data is very useful in criminal investigations (thanks other group :)). In the dataset, there are 6 existing types of glass and 2 main attributes. The glass types are building window float processed, building window non-float processed, vehicle window float processed, containers, tableware, and headlamps. The attributes are refractive index and chemical components, which can be further broken down into Sodium, Magnesium, Aluminum, Silicon, Potassium, Calcium, Barium, and Iron. We found the dataset, which consists of 214 elements, on Kaggle and hoped that it would be useful for the purposes of running KNN classification tests.

# ## Statistical Summary

# ### Descriptive Statistics
# 
# The following tables show summary statistics for each type of glass:

# In[2]:


g2 = glass.drop(['Type'],axis=1)
g3 = g2.groupby(['Type_Name'])


# #### Building Windows: Float Processed

# In[3]:


print(g3.get_group('bwfp ').describe().transpose())


# #### Building Windows: Non-Float Processed

# In[4]:


print(g3.get_group('bwnfp').describe().transpose())


# #### Vehicle Windows: Float Processed

# In[5]:


print(g3.get_group('vwfp ').describe().transpose())


# #### Containers

# In[6]:


print(g3.get_group('c').describe().transpose())


# #### Tableware

# In[7]:


print(g3.get_group('t ').describe().transpose())


# #### Headlamps

# In[8]:


print(g3.get_group('h').describe().transpose())


# ### Scatter Matrix

# In[9]:


from sklearn.model_selection import train_test_split
from matplotlib import cm
from pandas.plotting import scatter_matrix

# separate out data
X = glass[['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']]
y = glass['Type']

# set data splits
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.75, random_state=0)

# plot matrix
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)


# ## Classification Summary
# 
# The following section is an updated version of the code which was originally submitted online.
# After the presentation, I realized that I messed several things up while I was testing the KNN classification.
# As a result, this will look much different from what it did before.
# 
# In this section, I am attempting to differentiate between a classifier measured with euclidean distance and a classifier measured with manhattan distance. When I originally ran the tests, I used uniform weights vs. distance weights, which is how I came up with 4-nn as the best division to make for this algorithm. In the current tests, I found that my results corroborated with those of the other group who used the glass dataset.

# In[10]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ### Testing with Euclidean Distance

# In[11]:


knn = KNeighborsClassifier(n_neighbors = 3, metric="euclidean")

# Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
knn.score(X_test, y_test)

y_pred = knn.predict(X_test)

# added in after the fact
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix for Euclidean')
print(cm)

# plotting the confusion matrix
fig, ax = plt.subplots()
cbar_ax = fig.add_axes([.92, .3, .02, .4])
sns.heatmap(cm, ax=ax,annot=True,cmap="Greens", cbar_ax=cbar_ax)
ax.set_title('Confusion Matrix for Euclidean Glass Classifier')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.xaxis.set_ticklabels(['BWFP','BWNFP','VWFP','C','T','H']);
ax.yaxis.set_ticklabels(['BWFP','BWNFP','VWFP','C','T','H']);


# ### Testing with Manhattan Distance

# In[12]:


knn = KNeighborsClassifier(n_neighbors = 3, metric="manhattan")

# Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
knn.score(X_test, y_test)

y_pred = knn.predict(X_test)

# added in after the fact
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix for Manhattan')
print(cm)

# plotting the confusion matrix
fig, ax = plt.subplots()
cbar_ax = fig.add_axes([.92, .3, .02, .4])
sns.heatmap(cm, ax=ax,annot=True,cmap="Greens", cbar_ax=cbar_ax)
ax.set_title('Confusion Matrix for Manhattan Glass Classifier')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.xaxis.set_ticklabels(['BWFP','BWNFP','VWFP','C','T','H']);
ax.yaxis.set_ticklabels(['BWFP','BWNFP','VWFP','C','T','H']);


# From the matrices presented into this section, I have found that Manhattan distance proved to be a better metric for classifying the glass dataset than Euclidean. While the general pattern for prediction remained the same between the two distance metrics, it appears that the ratio between actual versus predicted was higher for Manhattan distance than Euclidean.

# ## Analysis of Accuracy
# 
# Because Manhattan distance proved to be a better metric for measuring the data, I have continued using this metric for the remaining analysis portions of this report.
# 
# ### Accuracy with different k values
# 
# In this section, I will be analyzing how different k values affect the accuracy of correctly predicting types of glass.

# In[18]:


scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, metric="manhattan")
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('Accuracy for Manhattan')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])
plt.show()


# From the above graph, one can see that the accuracy for correctly predicting glass types peaks at around 3-knn. After that, the accuracy follows a downward trend.

# ### Accuracy related to different partitions of data
# 
# In this portion, I will be analyzing how different partitions of the data affect the overall accuracy of predicting glass type. I will be using the assumptions that Manhattan distance is the best metric and that 3-knn is the best number of neighbors to use.

# In[16]:


# manhattan measures
knn = KNeighborsClassifier(n_neighbors=3,metric="manhattan")
plt.figure()
for s in t:
    scores = []
    for i in range(1, 1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
plt.title('Accuracy for Manhattan')
plt.show()


# As one can see from the above chart, the accuracy for the predicting glass type tends to increase as the proportion of data contained in the training set increases. The increase for this also seems to be steady. In this case, even taking a training set proportion of 60% or 70% could result in a decently high accuracy.

# ## Data Leakage Problem & Glass Data
# 
# Data leakage refers to when a partition of testing data uses replicate values from a training dataset. In this case, the algorithm we are using randomly partitions the data into testing and training sets. It separates the data out, preventing duplicate values from being in the testing set. Therefore, the data leakage problem is prevented in our implementation of KNN.
# 
# ## Conclusion
# 
# As a result of the analysis which was given in this notebook, I have found that Manhattan distance was a better metric for measuring the glass dataset. In addition, the best K to use for the dataset is 3 nearest neighbors. The best proportion could range from 50% to 70% of the data.
# 
# In order to improve the accuracy of this model, it would probably be best to first increase the size of the set. Perhaps one could look at the original set from which the Kaggle set was taken. In addition, it might be useful to run other supervised learning algorithms to see if the accuracy could be improved in that respect.
