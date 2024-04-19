## Imports

import numpy as np
import pandas as pd
import math

# Safelog implemention to prevent -inf

def safelog(x):
    return np.log(x + 1e-100)

# Score function

def score(data):
    global K,priors,means,deviations
    score = np.stack([[np.sum(- 0.5 * safelog(2 * math.pi * deviations[:,c]**2) 
                         - 0.5 * (data[i] - means[:,c])**2 / deviations[:,c]**2) 
                         + safelog(priors[c])
                         for c in range(K)] for i in range(data.shape[0])])
    return score

# Importing data sets

data_set = np.genfromtxt("hw01_images.csv", delimiter = ',')
y = np.genfromtxt("hw01_labels.csv", dtype = 'int')

# Training set and test set

training_set = data_set[:200,:]
test_set = data_set[200:400,:]

# Number of classes
K = np.max(y)

# Prior probabilities of classes
priors = [np.mean(y[0:200] == c+1) for c in range(K)]

# Labels of the data points in training set and test set

y_training = y[0:200]
y_test = y[200:400]

# Means and deviations of all data points

means = np.stack([[np.mean(training_set[:,i][y_training == (j+1)]) for i in 
                   range(data_set.shape[1])] for j in range(K)], axis = 1)
deviations = np.stack([[np.sqrt(np.var(training_set[:,i][y_training == (j+1)])) 
                        for i in range(data_set.shape[1])] for j in range(K)], 
                      axis = 1)

# Scores of training set and the corresponding confusion matrix
train_score = score(training_set)
train_predicted = np.argmax(train_score, axis = 1) + 1

confusion_matrix1 = pd.crosstab(y_training,train_predicted, 
                                rownames = ['y_train'], colnames = ['y_hat'])
print(confusion_matrix1)

# Scores of test set and the corresponding confusion matrix
test_score = score(test_set)
test_predicted = np.argmax(test_score, axis = 1) + 1

confusion_matrix2 = pd.crosstab(y_test,test_predicted, 
                                rownames = ['y_test'], colnames = ['y_hat'])
print(confusion_matrix2)

