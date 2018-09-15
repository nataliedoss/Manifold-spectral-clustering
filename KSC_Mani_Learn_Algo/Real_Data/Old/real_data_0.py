################################################################################
################################################################################
################################################################################
# Real data 


#################################################################################
#################################################################################
#################################################################################
# Modules. Import these to each file that uses the algorithm.
import numpy as np
import random
# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# sklearn. Maybe do import sklearn * or something?
import sklearn.cluster
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
# From scipy for kmeans:
import scipy
from scipy import cluster
# Other
import itertools # For permutations
import csv # To make csv's
import time # To time things
rng = np.random.RandomState(0) # Random number generator

#################################################################################
# My modules
import sys
sys.path.insert(0, "/Users/nataliedoss/Dropbox/Work/Research/Manifold_Learning/Code/Algorithm_Python")
import Refinement_Algorithm_Package
from Refinement_Algorithm_Package import refinement as r


################################################################################
################################################################################
################################################################################
# To use:
start_time = time.time()
A = r.create_adjacency_matrix(n, X, rho)
sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='kmeans')
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
error_init = np.sum(sigma_tilde_final != sigma)
# Refine:
sigma_hat_nodr = r.refine(X, n, p, k, rho,
                        sigma, sigma_tilde, perms,
                        r.reduce_dimension_none, nc, m_est,
                        r.estimate_manifold_avg, nb_size_mani, coeff_mani_est,
                        r.create_X_new, nb_size_X, coeff_X_new,
                        K)
sigma_hat_dr = r.refine(X, n, p, k, rho,
                        sigma, sigma_tilde, perms,
                        r.reduce_dimension_llp, nc, m_est,
                        r.estimate_manifold_avg, nb_size_mani, coeff_mani_est,
                        r.create_X_new, nb_size_X, coeff_X_new,
                        K)


print (100*np.sum(sigma_tilde_final != sigma) + 0.0)/n
print (100*np.sum(sigma_hat_nodr != sigma) + 0.0)/n
print (100*np.sum(sigma_hat_dr != sigma) + 0.0)/n

print("--- %s seconds ---" % (time.time() - start_time))


################################################################################
################################################################################
################################################################################
# Heart dataset:
data = np.loadtxt('SPECTF.train.txt', delimiter=',')
X = data[:,1:44]
sigma = data[:,0]

# Set data parameters and initialize:
n = len(X)
k = 2
p = len(X[0])
rho = 2000*p
A = r.create_adjacency_matrix(n, X, rho)
sigma_tilde = r.create_sc_sigma_tilde(A, n, k)
perms = np.array(list(itertools.permutations(range(0,k))))
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)


# Parameters
dimension_reduction_method = r.reduce_dimension_svd
nc = 10
m_est = 5
manifold_estimation_method = r.estimate_manifold_kmeans
nb_size_mani = np.repeat(2,k)
coeff_mani_est = A
X_new_creation_method = r.create_X_new
nb_size_X = 1
coeff_X_new = A
K = 3


sigma_1 = r.refine(X, n, p, k, rho, sigma, sigma_tilde, perms,
                        dimension_reduction_method, nc, m_est,
                        manifold_estimation_method, nb_size_mani, coeff_mani_est,
                        X_new_creation_method, nb_size_X, coeff_X_new, K)
sigma_2 = r.refine(X, n, p, k, rho, sigma, sigma_1, perms,
                        dimension_reduction_method, nc, m_est,
                        manifold_estimation_method, nb_size_mani, coeff_mani_est,
                        X_new_creation_method, nb_size_X, coeff_X_new, K)

error_init = np.sum(sigma != sigma_tilde_final)
error_1 = np.sum(sigma != sigma_1)
error_2 = np.sum(sigma != sigma_2)

print error_init
print error_1
print error_2







################################################################################
################################################################################
################################################################################
# Wine dataset:
data = np.loadtxt('wine.txt', delimiter=',')
X = data[:,1:14]
sigma = data[:,0] - 1

# Set data parameters and initialize:
n = len(X)
k = 3
p = len(X[0])
rho = 30000*p
A = r.create_adjacency_matrix(n, X, rho)
sigma_tilde = r.create_sc_sigma_tilde(A, n, k)
perms = np.array(list(itertools.permutations(range(0,k))))
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)

# Parameters
dimension_reduction_method = r.reduce_dimension_svd
nc = 10
m_est = 5
manifold_estimation_method = r.estimate_manifold_kmeans
nb_size_mani = np.repeat(2,k)
coeff_mani_est = A
X_new_creation_method = r.create_X_new
nb_size_X = 1
coeff_X_new = A
K = 3

# Loop for this iteration:
iterations = 10
error_array = np.empty(iterations)
sigma_t = sigma_tilde_final
for iter in range(iterations):
    sigma_new = r.refine(X, n, p, k, rho, sigma, sigma_t, perms,
                        dimension_reduction_method, nc, m_est,
                        manifold_estimation_method, nb_size_mani, coeff_mani_est,
                        X_new_creation_method, nb_size_X, coeff_X_new, K)
    error_array[iter] = np.sum(sigma != sigma_new)
    sigma_t = sigma_new


error_init = np.sum(sigma != sigma_tilde_final)
print error_init
print error_array




################################################################################
################################################################################
################################################################################
# Glass. 
data = np.loadtxt('glass.txt', delimiter=',')
X = data[:,0:9]
labels = data[:,10]
sigma = labels.copy()
# Do this in a quicker way: 
sigma[sigma == 1] = 0
sigma[sigma == 2] = 1
sigma[sigma == 3] = 2
sigma[sigma == 5] = 3
sigma[sigma == 6] = 4
sigma[sigma == 7] = 5

# Set data parameters and initialize:
n = len(X)
k = 3
p = len(X[0])
rho = 698700
A = r.create_adjacency_matrix(n, X, rho)
sigma_tilde = r.create_sc_sigma_tilde(A, n, k)
perms = np.array(list(itertools.permutations(range(0,k))))
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)

# Parameters
dimension_reduction_method = r.reduce_dimension_svd
nc = 5
m_est = 4
manifold_estimation_method = r.estimate_manifold_kmeans
nb_size_mani = np.repeat(2,k)
coeff_mani_est = A
X_new_creation_method = r.create_X_new
nb_size_X = 1
coeff_X_new = A
K = 3



# Loop for this iteration:
iterations = 10
error_array = np.empty(iterations)
sigma_t = sigma_tilde_final
for iter in range(iterations):
    sigma_new = r.refine(X, n, p, k, rho, sigma, sigma_t, perms,
                        dimension_reduction_method, nc, m_est,
                        manifold_estimation_method, nb_size_mani, coeff_mani_est,
                        X_new_creation_method, nb_size_X, coeff_X_new, K)
    error_array[iter] = np.sum(sigma != sigma_new)
    sigma_t = sigma_new


error_init = np.sum(sigma != sigma_tilde_final)
print error_init
print error_array






















'''

################################################################################
################################################################################
################################################################################
# MNIST
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
data = mnist.data
labels = mnist.target
X = data[np.logical_or(labels==0, labels==1),]
sigma = labels[np.logical_or(labels==0, labels==1) ]
# Subset more:
subset = 1000
X = X[0:subset,]
sigma = sigma[0:subset]

# Assign parameters:
n = len(X)
k = 2 # MAKE GENERAL
perms = np.array(list(itertools.permutations(range(0,k))))
p = len(X[0])
rho = 5000*p # For Gaussian kernel. 
nb_size_manifolds = np.repeat(2,k)
nb_size_X = 5
lam = .01
K = 7
# Dimension reduction:
nc = 2
A = create_adjacency_matrix(n, X, rho)
sigma_tilde = create_sc_sigma_tilde(A, n, k)
X_reduced = reduce_dimension(X, sigma_tilde, nc)


############################################################
# Run the algorithm
start_time = time.time()
errors_mnist = algorithm(X, n, k, sigma, perms, p, rho, nb_size_manifolds, nb_size_X, K, create_sc_sigma_tilde)
errors_mnist_dr = algorithm(X_reduced, n, k, sigma, perms, p, rho, nb_size_manifolds, nb_size_X, K, create_sc_sigma_tilde)
print("--- %s seconds ---" % (time.time() - start_time))
print errors_mnist





################################################################################
################################################################################
################################################################################
# Newsgroups
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
categories = ['alt.atheism', 'talk.religion.misc']
newsgroups = fetch_20newsgroups(categories=categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups.data)
X = vectors.toarray()
sigma = newsgroups.target


# Assign parameters:
n = len(X)
k = 2
perms = np.array(list(itertools.permutations(range(0,k))))
p = len(X[0])
rho = 10000*p # For Gaussian kernel. 
nb_size_manifolds = np.array((3, 3))
nb_size_X = 3
lam = .01
K = 7
K

# Takes about 7 minutes to run
start_time = time.time()
errors_newsgroups = algorithm(X, n, k, sigma, perms, p, rho, nb_size_manifolds, nb_size_X, K, create_sc_sigma_tilde)
print("--- %s seconds ---" % (time.time() - start_time))
print errors_newsgroups




############################################################
############################################################
############################################################
# Store the results for all datasets and algorithms:

# Format the data for a csv:
errors_data = np.column_stack((errors_mnist, errors_newsgroups, errors_glass))

# Row and column names for the csv:
rownames = ["SC", "Manifold + X", "Manifold + X Average", "Manifold + X Kernel", "Manifold Average + X", "Manifold Average + X Average", "Manifold Average + X Kernel", "Manifold Kernel + X", "Manifold Kernel + X Average", "Manifold Kernel + X Kernel"]
colnames = ["Algorithm", "MNIST", "Newsgroups", "Glass"]

# Create the csv of the data:
with open('errors_data_next.csv', 'wb') as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(colnames)
    for row_title, data_row in zip(rownames, errors_data):
        writer.writerow([row_title] + data_row.tolist())

# Create the csv of the caption in LaTeX:
# Add the parameters of other datasets here!
with open('errors_data_params_next.csv', 'wb') as f:
    writer = csv.writer(f, delimiter=";")   
    writer.writerow( [", MNIST: n = %i; k = %i; p = %i; rho = %i; K = %i; lambda = %f; Neighborhood sizes are %i and %i" %(n, k, p, rho, K, lam, nb_size_manifolds[0], nb_size_manifolds[1])] )
















'''
