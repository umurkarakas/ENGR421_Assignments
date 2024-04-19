import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy.linalg as linalg

## importing dataset
X = np.genfromtxt("hw06_data_set.csv",delimiter = ",", skip_header=1)

## plot function
def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["red", "limegreen", "orange", "rebeccapurple", "dodgerblue"])
    plt.subplots(figsize=(8,8))
    plt.xlim(-6.2,6.2)
    plt.ylim(-6.2,6.2) 
    for c in range(K):
        plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 12, 
                 color = cluster_colors[c])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

## parameters and initialization of matrices
s = 1.25
N = len(X)
K = 5
B = np.zeros((N,N))
D = np.zeros((N,N))

## adjacency matrix
for i in range(N):
    for j in range(i,N):
        if(np.linalg.norm(X[i]-X[j]) <= s):
            B[i][j] = 1
            B[j][i] = 1
np.fill_diagonal(B, 0.0)

## sum of adjacencies matrix
for i in range(N):
    D[i][i] = sum(B[i])
    

## plot of adjacent points
plt.subplots(figsize=(8,8))
plt.xlim(-6.2,6.2)
plt.ylim(-6.2,6.2)
for i in range(N):
    for j in range(i,N):
        if(B[i][j] == 1):
            arr = np.array([X[i], X[j]])
            plt.plot(arr[:,0], arr[:,1], "k-", marker = 'o', ms = 4)
plt.show()

## normalized laplacian matrix
L = np.identity(N) - np.matmul(np.matmul(linalg.inv(np.sqrt(D)), B), linalg.inv(np.sqrt(D)))

## 5 smallest eigenvalues and their corresponding eigenvectors
eigenvalues,eigenvectors = linalg.eigh(L, subset_by_index = [0,4])

## Z matrix
Z = eigenvectors

## initial centroids and memberships
centroids = np.array([Z[84], Z[128], Z[166], Z[186], Z[269]])
memberships = np.argmin(spa.distance_matrix(Z,centroids), axis = 1)


## k-means clustering algorithm
while True:
    old_centroids = centroids
    centroids = np.vstack([np.mean(Z[memberships == k], axis = 0) for k in range(K)])
    old_memberships = memberships
    memberships = np.argmin(spa.distance_matrix(Z,centroids), axis = 1)
    if np.alltrue(centroids == old_centroids):
        break
  
## plot of clusters
plot_current_state(centroids, memberships, X)

    

