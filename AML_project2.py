
# coding: utf-8

# # Applied Machine Learning Project 2

# The data which will be used in this report relates to fall data collected for elderly patients in Chinese hospitals in 2017. The goal of the collection was to use machine learning algorithms in order to better predict if a fall occurred based on factors which include monitoring time (TIME), sugar level (SL), EEG monitoring rate (EEG), blood pressure (BP), heart rate (HR), and blood circulation (CIRCULATION). Other activities which the elderly patient could have been doing include sitting, running, standing, walking, and experiencing cramps.

# In[1]:


import pandas as pd

from knn import _knn
from logistic import _logistic
from SVM import _SVM

falldetection = pd.read_csv('data/falldetection.csv')
falldetection.head()

# create a mapping from activity label value to activity name to make results easier to interpret
lookup_fall_type = dict(zip(falldetection.ACTIVITY.unique(), falldetection.NAME.unique()))
print(lookup_fall_type)

X = falldetection[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']]
y = falldetection['ACTIVITY']


# ### Descriptive Statistics for Activities

# In[11]:


grouped = falldetection.groupby(['ACTIVITY'])
print('Falling')
print(grouped.get_group(3).describe().transpose())


# In[14]:


print('Sitting')
print(grouped.get_group(2).describe().transpose())


# In[15]:


print('Cramps')
print(grouped.get_group(4).describe().transpose())


# In[16]:


print('Running')
print(grouped.get_group(5).describe().transpose())


# In[17]:


print('Standing')
print(grouped.get_group(0).describe().transpose())


# In[18]:


print('Walking')
print(grouped.get_group(1).describe().transpose())


# ## Comparison of KNN, Logistic Regression, and SVM
# 
# The specific algorithms which we will be comparing today are KNN, logistic regression, and support vector machines. Since KNN was covered in the previous project, I won't talk about it here.
# 
# **Logistic regression** deals with modeling the probability of either/or situations. In this case, we are seeking to model and predict the probability that an elderly individual fell when given different associated attributes. 
# 
# **SVM** is a supervised machine learning algorithm that seeks to classify data by using support vectors. The margin between these vectors is called a hyperplane, which serves as a soft-margin to differentiate between two classes. In this case, the person fell or the person did not fall.

# ### Algorithm 1: KNN

# In[3]:


_knn(X, y)


# From our accuracy chart, it would appear that using 4-nearest neighbors produces the best results for predicting the activity classification.

# ### Algorithm 2: Logistic Regression

# In[2]:


_logistic(X, y, lookup_fall_type)


# When we compare the confusion matrix from KNN with the confusion matrix from logistic regression, we can see that logistic regression proves to better predict whether an activity that occurred was a fall or was not a fall. However, KNN is better at predicting the type of activity that occurred (not just related to fall or not fall).

# ### Algorithm 3: SVM

# In[2]:


_SVM(X, y)


# Since logistic regression appears to be a better predictor than KNN for the choice of fall/not fall, I want to look at how SVM compares. In this cases, one can see that the linear implementation of SVM does not hold up to logistic regression in terms of both classification by fall/not fall and by general classification among the different activities. However, SVM does do better than KNN when classifying simply by fall/not fall criteria. On the other hand, KNN is better at classifying by general activities than both logistic regression and SVM. This is because logistic regression and SVM both tended to classify non-fall related activities as "Standing" whereas KNN tended to have a more even distribution of wrongly classified activities.

# ### Concluding Remarks

# Based on this analysis, I present logistic regression as the best fall/not fall predictor for the data which was presented. SVM closely follows logistic regression in this classification, and KNN comes up last in the lineup. However if an analyst wished to, instead, consider all of the activities that the patient could have been doing, an approach using KNN would be better for classification than either logistic regression or SVM.
