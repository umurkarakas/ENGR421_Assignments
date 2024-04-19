import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy.stats as stats

X = np.genfromtxt("hw05_data_set.csv",delimiter = ",", skip_header=1)

## means and covariances used to generate data in the csv
initial_means = np.array([[-2.5,-2.5], [2.5,2.5], [2.5,-2.5], [0.0,0.0], [-2.5,2.5]])
initial_covariances = np.array([[[0.8,-0.6],[-0.6,0.8]],[[0.8,-0.6],[-0.6,0.8]],[[0.8,0.6],[0.6,0.8]],[[1.6,0.0],[0.0,1.6]],[[0.8,0.6],[0.6,0.8]]])
    
#initial centroids
centroids = np.genfromtxt("hw05_initial_centroids.csv",delimiter = ",")

#number of clusters and data points
K = centroids.shape[0]
N = X.shape[0]

#initial memberships and one-hat-encoding of memberships
memberships = np.argmin(spa.distance_matrix(centroids, X), axis = 0)
Z = np.zeros((N,K)).astype(int)
Z[range(N), memberships] = 1

#initial covariances and priors
covariances = np.stack([np.cov(np.transpose(X[memberships == i])) for i in range(K)])
priors = np.stack([(memberships == i).sum() / N for i in range(K)])

iteration = 100

for a in range(iteration):
    gaussians = np.stack([stats.multivariate_normal.pdf(X, mean = centroids[k], cov = covariances[k]) for k in range(K)])
    success = np.transpose(np.stack([stats.multivariate_normal.pdf(X, mean = centroids[k], cov = covariances[k]) for k in range(K)])) * priors
    for i in range(N):
        success[i] /= np.sum(success[i])   
    centroids = np.stack([np.matmul(success[:,k], X) / np.sum(success[:,k]) for k in range(K)])
    covariances = np.stack([np.sum(np.stack([success[i][k] * np.matmul(np.transpose(np.array([X[i]-centroids[k]])), np.array([(X[i] - centroids[k])])) for i in range(N)]), axis = 0) / np.sum(success[:,k]) for k in range(K)])
    priors = np.stack([np.sum(success[:,k]) / N for k in range(K)])

#final memberships of each data point
memberships = np.argmax(success, axis = 1)

#plot
x1_interval = np.linspace(-6, +6, 41)
x2_interval = np.linspace(-6, +6, 41)
x1, x2 = np.meshgrid(x1_interval, x2_interval)
cluster_colors = np.array(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])
fig = plt.subplots(figsize=(8,8))
plt.xlim(-6.2,6.2)
plt.ylim(-6.2,6.2)
for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 12, 
             color = cluster_colors[c])
    plt.contour(x1, x2, stats.multivariate_normal.pdf(np.dstack((x1,x2)), mean = initial_means[c], cov = initial_covariances[c]), levels = [0.05], linestyles = ["dashed"], colors = "k")
    plt.contour(x1, x2, stats.multivariate_normal.pdf(np.dstack((x1,x2)), mean = centroids[c], cov = covariances[c]), levels = [0.05], linestyles = ["solid"], colors = "k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

