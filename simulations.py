################################################################################
################################################################################
################################################################################
# Simulations for algorithm

# Modules
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import itertools # For permutations
import time # To time things
rng = np.random.RandomState(0) # Random number generator

from refinement import * 
from create_sim_data import * 



# Parameters
# For the simulations, we have both m_create (used to create the data) and m (used to estimate the manifold)
n = 1000
k = 2
p = 10
rho = p/4.0

# Sigma and its permutations:
sigma = np.repeat(np.arange(0,k), [n/k]*k)
perms = np.array(list(itertools.permutations(range(0,k))))

# Parameters for data generation:
m_create = 2  # Full (ambient + true) dimension. 
epsilon = 1.0 # Should be a decimal.
d = 4*np.sqrt(k)
b = 2
n_turns = 1.5 # For Swiss rolls
nc = 20 # Must be much smaller than each estimated cluster size
m = 2
nb_size_mani = np.repeat(1,k) # No averaging, single points for mani est
coeff_mani_est = np.repeat(1,n**2).reshape(n,n)
nb_size_X = 10
coeff_X_new = np.repeat(1,n**2).reshape(n,n)
K = 20 # For KNN in refinement
#create_manifolds = create_circles
#X_new_creation_method = create_X_new


# Plot the data

# Create data and plot
mani = create_circles(n, k, m, d, b, n_turns, sigma)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(mani[:,0], mani[:,1])
plt.show()


# 
X = create_simulated_data(n, p, k, epsilon, m_create, d, b, n_turns, sigma, create_circles)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:,0], X[:,1])
plt.show()





################################################################################
################################################################################
################################################################################
# Test:
start_time = time.time()
X = create_simulated_data(n, p, k, epsilon, m_create, d, b, n_turns, sigma, create_circles)
sigma_estimates = initialize_and_refine(X, n, p, m, k, rho, sigma, nc, nb_size_mani, coeff_mani_est, create_X_new, nb_size_X, coeff_X_new, K)
print("--- %s seconds ---" % (time.time() - start_time))

print(np.sum(sigma_estimates[0] != sigma))
print(np.sum(sigma_estimates[1] != sigma))
print(np.sum(sigma_estimates[2] != sigma))
