################################################################################
################################################################################
################################################################################
# Gaussian mixture model simulations:
# Testing how spectral clustering does with dimension of data
# distance between means
# and different kernels


#################################################################################
#################################################################################
#################################################################################
# Modules

# Basic
import numpy as np
import random

# Plotting
import matplotlib.pyplot as plt

# sklearn
import sklearn.cluster
from sklearn.cluster import KMeans

# Other
import itertools # For permutations
import time # To time things
rng = np.random.RandomState(0) # Random number generator



#################################################################################
#################################################################################
#################################################################################
# Create Gaussian mixture data with k = 2

k = 2
n = 100
p = 2
mu0 = [0]*p
mu1 = [3]*p
sigma = 0.5 # noise level
cov = np.zeros((p,p), float)
np.fill_diagonal(cov, sigma)

# Need to pick the means at random. Improve this.
num_1 = np.random.binomial(n, 0.5, size = None)
mean_0 = np.array(np.repeat(mu0, n - num_1)).reshape(n - num_1, p)
mean_1 = np.array(np.repeat(mu1, num_1)).reshape(num_1, p)
mean = np.concatenate((mean_0, mean_1), axis = 0)


# Create the final data
Z = np.random.multivariate_normal(np.repeat(0, p), cov, n).T
Z = Z.reshape(n,p)
X = mean + Z

# Check the data with a plot
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.show()



#################################################################################
#################################################################################
#################################################################################
# Functions to create various adjacency matrices

def create_A_covariance(n, X, rho):
    return np.dot(X, X.T)

def create_A_euclidean(n, X, rho):
    A1 = np.zeros((n,n))
    A3 = np.zeros((n,n))
    for i in range(0,n):
        A1[i,] = (np.linalg.norm(X[i,]))**2
        A3[i,] = (np.linalg.norm(X[i,]))**2

    A2 = np.dot(X, X.T)
    A = A1 + A3.T - (2*A2)
    #np.fill_diagonal(A, 0)
    return A

def create_A_gaussian(n, X, rho):
    A1 = np.zeros((n,n))
    A3 = np.zeros((n,n))
    for i in range(0,n):
        A1[i,] = (np.linalg.norm(X[i,]))**2
        A3[i,] = (np.linalg.norm(X[i,]))**2

    A2 = np.dot(X, X.T)
    A = A1 + A3.T - (2*A2)
    A = np.exp(-A/rho)
    #np.fill_diagonal(A, 0)
    return A



#################################################################################
#################################################################################
#################################################################################
# Create the actual matrices
# Also do this for Laplacians!

rho = 1
A_cov = create_A_covariance(n, X, rho)

A_cov


#################################################################################
#################################################################################
#################################################################################
# Actual spectral clustering, but just with adjacency matrices

sc_cov = sklearn.cluster.spectral_clustering(A_cov, n_clusters = k,
                                               assign_labels = 'kmeans')





