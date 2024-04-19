## imports
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## data import   
data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",", skip_header=True)

## training set and test set
X = data_set[:,0]
y = data_set[:,1]

y_train = y[:150]
y_test = y[150:]

x_train = X[:150]
x_test = X[150:]

## parameters
K = np.max(y)
N = X.shape[0]

minval = min(X)
maxval = max(X)

N_training = len(y_train)
N_test = len(y_test)

## learn tree algorithm
def learnTree(x_train, y_train, P):
    N_train = len(x_train)
    node_indices = {}
    is_terminal = {}
    need_split = {}
    node_splits = {}
    node_means = {}    
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True
    
    while True:
        split_nodes = [key for key, value in need_split.items() if value == True]
        if len(split_nodes) == 0:
            break
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_mean = np.mean(y_train[data_indices])
            
            ## if node has less than 25 data points, converting into terminal node
            if x_train[data_indices].size <= P:
                is_terminal[split_node] = True
                node_means[split_node] = node_mean
                
            else:
                is_terminal[split_node] = False
                unique_values = np.sort(np.unique(x_train[data_indices]))
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values)-1)])/2
                split_scores = np.repeat(0.0,len(split_positions))
                for s in range(len(split_positions)):
                    left_indices = data_indices[x_train[data_indices] < split_positions[s]]
                    right_indices = data_indices[x_train[data_indices] >= split_positions[s]]
                    sum_error = 0
                    if len(left_indices)>0:
                        sum_error += np.sum((y_train[left_indices] - np.mean(y_train[left_indices])) ** 2)
                    if len(right_indices)>0:
                        sum_error += np.sum((y_train[right_indices] - np.mean(y_train[right_indices])) ** 2)
                    split_scores[s] = sum_error/(len(left_indices)+len(right_indices))
                    
                ## if len == 1, it is unique
                if len(unique_values) == 1 :
                    is_terminal[split_node] = True
                    node_means[split_node] = node_mean
                    continue
                best_split = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_split
                
                # creating left node using the selected split
                left_indices = data_indices[(x_train[data_indices] < best_split)]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node]  = False
                need_split[2 * split_node] = True

                # creating right node using the selected split
                right_indices = data_indices[(x_train[data_indices] >= best_split)]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1]  = True
    return node_splits,node_means,is_terminal

def predict(x, node_splits, node_means, is_terminal):
    index = 1 ## root node
    while True:
        if is_terminal[index] == True:
            return node_means[index]
        if x > node_splits[index]:
            index = index * 2 + 1 ## right child
        else:
            index = index * 2 ## left child

P=25
node_splits,node_means,is_terminal = learnTree(x_train,y_train,P)
pred = np.stack([predict(x,node_splits,node_means,is_terminal) for x in x_test])
data_interval = np.linspace(minval - 0.1,maxval + 0.1,1001)
fig = plt.figure(figsize=(10,4))
plt.plot(x_train,y_train,"b.", alpha = 0.4, label="training", ms = 8)
plt.plot(x_test,y_test,"r.", alpha = 1, label="test", ms = 8)
y_values = []
for i in range(len(data_interval)):
    y_values.append(predict(data_interval[i],node_splits,node_means,is_terminal))
plt.plot(data_interval,y_values,"k")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.title("P=25")
plt.legend(loc="upper left")
plt.show()

def rmse(y,pred):
    return np.sqrt(np.sum((y - pred) ** 2) / len(y))

print("RMSE is {:.4f}".format(rmse(y_test,pred)),"when P is {}".format(P))

x_values = []
rmse_values = []

for i in range(5,55,5): ## for P = 5,10,15,...,50
    x_values.append(i)
    node_splits,node_means,is_terminal = learnTree(x_train,y_train,i)
    pred = [predict(x,node_splits,node_means,is_terminal) for x in x_test]
    rmse_values.append(rmse(y_test,pred))

fig = plt.figure(figsize=(10,4))
plt.plot(x_values,rmse_values,"-ko", marker = 'o')
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.show()