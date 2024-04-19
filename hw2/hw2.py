## Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Safelog implemention to prevent -inf

def safelog(x):
    return np.log(x + 1e-100)

# Importing data sets

X = np.genfromtxt("hw02_images.csv", delimiter = ',')
Y = np.genfromtxt("hw02_labels.csv", delimiter = ',').astype(int)
W = np.genfromtxt("initial_W.csv", delimiter=',')
w0 = np.genfromtxt("initial_w0.csv", delimiter=',')

# Training set and test set

training_set = X[0:500]
test_set = X[500:1000]

# Number of classes and features

K = np.max(Y)
N = X.shape[0]

# One-of-K encoding

Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), Y - 1] = 1

# Labels of the data points in training set and test set

Y_training = Y_truth[0:500]
Y_test = Y_truth[500:1000]

# Learning parameters

eta = 0.0001
epsilon = 1e-3
max_iteration = 500

# Sigmoid function

def sigmoid(X,W,w0):
    return(1 / (1 + np.exp(-(np.matmul(X, W) + w0))))

# Gradient functions for W and w0

def gradient_W(X, y_truth, y_predicted):
    return(np.stack([-np.sum(np.repeat((y_truth[:,c] - y_predicted[:,c])[:, None], X.shape[1], axis = 1) * 
                               np.repeat((y_predicted[:,c])[:, None], X.shape[1], axis = 1) * 
                               np.repeat((1 - y_predicted[:,c])[:, None], X.shape[1], axis = 1) * 
                               X, axis = 0) for c in range(K)]).transpose())

def gradient_w0(y_truth, y_predicted):
    return(-np.sum((y_truth - y_predicted) * y_predicted * (1 - y_predicted), axis = 0))


iteration = 1
objective_values = []

# Iteration to learn W and w0

while iteration < max_iteration:
    
    Y_predicted = sigmoid(training_set, W, w0)
    
    objective_values = np.append(objective_values, 1/2 * np.sum((Y_training - Y_predicted)**2))

    W_old = W
    w0_old = w0
    W = W - eta * gradient_W(training_set, Y_training, Y_predicted)
    
    w0 = w0 - eta * gradient_w0(Y_training, Y_predicted)
    
    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break
    iteration = iteration + 1
    
# Error plot

plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# Confusion matrix for the training set

y_training_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(y_training_predicted, Y[0:500], rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)

print()

# Confusion matrix for the test set

y_test_predicted = np.argmax(sigmoid(test_set,W,w0), axis = 1) + 1
confusion_matrix = pd.crosstab(y_test_predicted, Y[500:1000], rownames = ['y_predicted'], colnames = ['y_test'])
print(confusion_matrix)




