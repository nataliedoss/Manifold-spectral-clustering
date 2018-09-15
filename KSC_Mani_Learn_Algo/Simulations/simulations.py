################################################################################
################################################################################
################################################################################
# Simulations for algorithm



#################################################################################
#################################################################################
#################################################################################
# Modules
import numpy as np
import random
# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# sklearn. Maybe do import sklearn * or something?
import sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
# Other
import itertools # For permutations
import csv # To make csv's
import time # To time things
rng = np.random.RandomState(0) # Random number generator


#################################################################################
# My modules
import sys
sys.path.insert(0, "/Users/nataliedoss/Dropbox/Work/Projects/KSC/Code/Algorithm_Python")
import Refinement_Algorithm_Package
from Refinement_Algorithm_Package import refinement as r
from Refinement_Algorithm_Package import create_sim_data as cr


################################################################################
# Manifold lists:
list_manifolds = [cr.create_circles, cr.create_curves, cr.create_lines]
names_manifolds = ["Circles", "Curves", "Lines"]
number_manifolds = len(list_manifolds)


################################################################################
################################################################################
################################################################################
# Parameters
# For the simulations, we have both m_create (used to create the data) and m (used to estimate the manifold)
n = 1000
k = 2
p = 100
rho = p/4

# Sigma and its permutations:
sigma = np.repeat(np.arange(0,k), [n/k]*k)
perms = np.array(list(itertools.permutations(range(0,k))))

# Parameters for data generation:
m_create = 2  # Manifold full (ambient + true) dimension. 
epsilon = 1.0 # Should be a decimal.
d = 2*np.sqrt(k)
b = 2
n_turns = 1.5 # For Swiss rolls

# Parameters for dimension reduction:
lam = .2 # For lasso
nc = 5 # Must be much smaller than each estimated cluster size
m = 40

# Parameters for manifold estimation:
#nb_size_mani = np.repeat(20,k)
nb_size_mani = np.repeat(1,k) # No averaging, single points for mani est
coeff_mani_est = np.repeat(1,n**2).reshape(n,n)

# Parameters for X_new:
nb_size_X = 10
coeff_X_new = np.repeat(1,n**2).reshape(n,n)

# Parameters for testing:
K = 7 # For KNN in refinement

# More:
create_manifolds = cr.create_circles
X_new_creation_method = r.create_X_new


################################################################################
################################################################################
################################################################################
# Testing SC with different strategies:
N = 100
sigma_lra_vec = np.empty((N,1))
sigma_sc_vec = np.empty((N,1))

for i in range(N):
    
    X = cr.create_simulated_data(n, p, k, epsilon, m_create, d, b, n_turns, sigma, create_manifolds)
    X_mk = r.lra(X, n, p, k)
    A = r.create_adjacency_matrix(n, X, rho)
    
    sigma_lra = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1).fit(X_mk).labels_   
    sigma_sc = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='kmeans')

    sigma_lra_final = r.perm_true(sigma, sigma_lra, n, k, perms)
    sigma_sc_final = r.perm_true(sigma, sigma_sc, n, k, perms)

    sigma_lra_vec[i] = np.sum(sigma_lra_final != sigma)
    sigma_sc_vec[i] = np.sum(sigma_sc_final != sigma)



print np.mean(sigma_lra_vec)

print np.mean(sigma_sc_vec)

# p = 5: 348.76 253.67

# p = 100: 353.79   408.02


################################################################################
################################################################################
################################################################################
# A test:
start_time = time.time()
X = cr.create_simulated_data(n, p, k, epsilon, m_create, d, b, n_turns, sigma, create_manifolds)
sigma_estimates = r.initialize_and_refine(X, n, p, m, k, rho, sigma, nc, nb_size_mani, coeff_mani_est, X_new_creation_method, nb_size_X, coeff_X_new, K)
print("--- %s seconds ---" % (time.time() - start_time))



print (100*np.sum(sigma_estimates[0] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[1] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[2] != sigma) + 0.0)/n




################################################################################
################################################################################
################################################################################
# Let's just do two versions for now, on circles:

start_time = time.time()
N = 2
create_manifolds = cr.create_circles
error_array = np.empty((N, 3)).reshape(N,3)
for i in range(N):
    X = cr.create_simulated_data(n, p, k, epsilon, m, d, b, n_turns, sigma, create_manifolds)
    A = r.create_adjacency_matrix(n, X, rho)
    sigma_tilde = r.create_sc_sigma_tilde(A, n, k)
    sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
    error_init = np.sum(sigma_tilde_final != sigma)
    # Refine:
    sigma_hat_dr = r.refine(X, n, p, k, rho, sigma, sigma_tilde, perms, r.reduce_dimension_svd, nc, m_est, r.estimate_manifold_kmeans, nb_size_mani, coeff_mani_est, r.create_X_new, nb_size_X, coeff_X_new, K)
    sigma_hat_nodr = r.refine(X, n, p, k, rho, sigma, sigma_tilde, perms, r.reduce_dimension_none, nc, m_est, r.estimate_manifold_avg, nb_size_mani, coeff_mani_est, r.create_X_new, nb_size_X, coeff_X_new, K)
    error_dr = np.sum(sigma_hat_dr != sigma)
    error_nodr = np.sum(sigma_hat_nodr != sigma)
    # Assign
    error_array[i,] = np.array((error_init, error_dr, error_nodr)).reshape(1,3)
print("--- %s seconds ---" % (time.time() - start_time))



print np.average(error_array[:,0])
print np.average(error_array[:,1])
print np.average(error_array[:,2])

# Note; X_new makes a big difference (all the difference) in non-dim reduced algo.
# E.g. after 50 sims:
#96.52
#7.36
#94.48







################################################################################
################################################################################
################################################################################
# Simple simulations, one row for each of the three manifolds:
N = 100
sims_mean = np.empty((number_outputs, number_manifolds))
# Loop:
for j in range(number_manifolds):
    create_manifolds = list_manifolds[j]
    nb_size_manifolds = list_nb_size_manifolds[j,]
    sims = np.zeros((N, number_outputs))
    for i in range(N):
        X = create_simulated_data(n, k, sigma, p, m, epsilon, d, b, n_turns, create_manifolds)
        test = algorithm(X, n, k, sigma, perms, p, m, epsilon, b, n_turns, rho, nb_size_manifolds, nb_size_X, K, lam, create_sc_sigma_tilde)
        sims[i,] = np.append(test[0], test[1].reshape(1, number_outputs-1))
    sims_mean[:,j] = (np.mean(sims[:,0]), np.mean(sims[:,1]), np.mean(sims[:,2]), np.mean(sims[:,3]), np.mean(sims[:,4]), np.mean(sims[:,5]), np.mean(sims[:,6]), np.mean(sims[:,7]), np.mean(sims[:,8]), np.mean(sims[:,9]), np.mean(sims[:,10]), np.mean(sims[:,11]), np.mean(sims[:,12]), np.mean(sims[:,13]), np.mean(sims[:,14]), np.mean(sims[:,15]), np.mean(sims[:,16]), np.mean(sims[:,17]), np.mean(sims[:,18]))


# Create csv file of results:
rownames = ["SC", "Manifold + X", "Manifold + X Average", "Manifold + X Kernel", "Manifold Average + X", "Manifold Average + X Average", "Manifold Average + X Kernel", "Manifold Kernel + X", "Manifold Kernel + X Average", "Manifold Kernel + X Kernel",  "Lasso: Manifold + X", "Lasso: Manifold + X Average", "Lasso: Manifold + X Kernel", "Lasso: Manifold Average + X", "Lasso: Manifold Average + X Average", "Lasso: Manifold Average + X Kernel", "Lasso: Manifold Kernel + X", "Lasso: Manifold Kernel + X Average", "Lasso: Manifold Kernel + X Kernel"]
colnames = ["Algorithm", names_manifolds[0], names_manifolds[1], names_manifolds[2]]
with open('sims_2.csv', 'wb') as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(colnames)
    for row_title, data_row in zip(rownames, sims_mean):
        writer.writerow([row_title] + data_row.tolist())
        
# Create csv file for caption in LaTeX:
with open('sims_2_params.csv', 'wb') as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow( [",N = %i; n = %i; k = %i; p = %i; epsilon = %f; d = %f; rho = %i; b = %i; K = %i; lambda = %f; Neighborhood sizes for first set of manifolds are %i and %i; Neighborhood sizes for second set of manifolds are %i and %i; Neighborhood sizes for third set of manifolds are %i and %i; Neighborhood size for averaged X is %i" %(N, n, k, p, epsilon, d, rho, b, K, lam, list_nb_size_manifolds[0,0], list_nb_size_manifolds[0,1], list_nb_size_manifolds[1,0], list_nb_size_manifolds[1,1], list_nb_size_manifolds[2,0], list_nb_size_manifolds[2,1], nb_size_X)] )



