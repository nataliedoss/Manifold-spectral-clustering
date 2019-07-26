#################################################################################
#################################################################################
#################################################################################
# Refinement Clustering Algorithm Functions



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
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
# Other
import itertools # For permutations
import csv # To make csv's
import time # To time things
rng = np.random.RandomState(0) # Random number generator




#################################################################################
#################################################################################
#################################################################################
# Several general functions used throughout:

# Return the permutation of sigma_est that's closest to sigma:
# Takes as input TKTK
def perm_true(sigma, sigma_est, n, k, perms):
    sigma_est_perms = np.empty((len(perms), n))
    for i in range(0, len(perms)):
        for j in range(0,k):
            sigma_est_perms[i,sigma_est==j] = perms[i,j]

    errors = np.empty(len(perms))
    for i in range(0, len(perms)):
        errors[i] = np.sum( sigma != sigma_est_perms[i,] )

    loc = np.where(errors == errors.min())[0]
    loc_min_error = loc[0] # In case there is more than one min
    return sigma_est_perms[loc_min_error,]


# Create adjacency matrices:
def create_adjacency_matrix(n, X, rho):
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

def create_adjacency_matrix(n, X, rho):
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

# Create the m-rank approximation of X:
def lra(X, n, p, m):
    U, D, V = np.linalg.svd(X, full_matrices=True)
    U_m = U[:,:m]
    D_m = np.diag(D[:m])
    V_m = V[:m,]
    return U_m.dot(D_m.dot(V_m))



###############################################################################
###############################################################################
###############################################################################
# Dimension Reduction.

# Local Linear Projection:
def reduce_dimension_llp(X, n, p, m, k, sigma_tilde, nc):
    all_manifolds_est = np.zeros((n,p))
    for l in range(k):
        # Reduce dimension to nc (number of clusters on single mani) and do kmeans:
        X_mani = X[sigma_tilde == l,]
        #X_nc = lra(X_mani, n, p, p) # Use this in k-means below if you do dimension reduction!!
        labels_nc = KMeans(n_clusters=nc, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1).fit(X_mani).labels_
        if m < np.bincount(labels_nc).min():
            m = m
        else:
            m = np.bincount(labels_nc).min()
        one_manifold_est = np.zeros((len(X_mani), p))
        for j in range(nc):
            X_local = X_mani[labels_nc == j,]
            mani_est_local = lra(X_local, n, p, m)
            one_manifold_est[labels_nc == j,] = mani_est_local
        all_manifolds_est[sigma_tilde == l, ] = one_manifold_est

    return all_manifolds_est

# No dimension reduction:
def reduce_dimension_none(X, n, p, m, k, sigma_tilde, nc):
    return X


###############################################################################
###############################################################################
###############################################################################
# Estimate the manifolds. 

# Estimate the manifolds via a weighted average:
def estimate_manifold_avg(X, n, p, k, sigma_tilde, nb_size, coeff):
    mani_est = np.zeros((n, p)) 
    for l in range(k):
        X_sub = X[sigma_tilde==l,]
        coeff_sub = coeff[sigma_tilde==l,]
        mani_sub = mani_est[sigma_tilde==l,]
        nbrs = NearestNeighbors(n_neighbors=int(nb_size[l]), algorithm='ball_tree').fit(X_sub)
        distances, indices = nbrs.kneighbors(X_sub)
        for i in range(0, len(X_sub)):
            mani_sub[i,] = np.array(np.average(X_sub[indices[i,],], axis=0, weights=coeff_sub[i,indices[i,]]))
        mani_est[sigma_tilde==l,] = mani_sub
        
    return mani_est, sigma_tilde



###############################################################################
###############################################################################
###############################################################################
# Create new points for testing (no dependence on manifolds):

# Estimate a new version of X via a weighted average:
def create_X_new(X, n, p, nb_size_X, coeff):
    X_new = np.zeros((n,p))
    nbrs = NearestNeighbors(n_neighbors=int(nb_size_X), algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    for i in range(0, n):
        X_new[i,] = np.array(np.average(X[indices[i,],], axis=0, weights=coeff[i,indices[i,]]))
    return X_new


###############################################################################
###############################################################################
###############################################################################
# Test

# Nearest neighbors testing function. In the event of a tie, it picks the first.
# For this function, sigma_centers must match mani_est.
def test(X, mani_est, k, sigma_centers, K):
    n = len(X)
    nc_full = len(mani_est)
    distance = np.zeros((n,nc_full)).reshape(n, nc_full)
    for i in range(n):
        for j in range(nc_full):
            distance[i,j] = np.linalg.norm(X[i,] - mani_est[j,])

    sigma_hat = np.empty(n)

    for i in range(n):
        ind = np.argpartition(distance[i,], K)[:K]
        nbs = sigma_centers[ind]
        counts = np.empty(k)
        for j in range(k):
            counts[j] = np.sum(nbs == j)
        # In the event of a tie, we just take the first one:
        sigma_hat[i] = np.where(counts == counts.max())[0][0]

    return sigma_hat


################################################################################
################################################################################
################################################################################
# The full refinement procedure with the option to reduce the dimension: 
    
def refine(X, n, p, m, k, rho,
           sigma, sigma_tilde, perms,
           dimension_reduction_method, nc, 
           manifold_estimation_method, nb_size_mani, coeff_mani_est,
           X_new_creation_method, nb_size_X, coeff_X_new,
           K):

    # Reduce dimension: 
    X_dr = dimension_reduction_method(X, n, p, m, k, sigma_tilde, nc)

    # Estimate manifold using dimension reduced data:
    mani_estimate = manifold_estimation_method(X_dr, n, p, k, sigma_tilde, nb_size_mani, coeff_mani_est)
    mani_est = mani_estimate[0]
    sigma_mani = mani_estimate[1]
    
    # Create X_new: NOT using dimension reduced version of X to average right now:
    X_new = create_X_new(X, n, p, nb_size_X, coeff_X_new)
    
    # Test:
    sigma_hat = test(X_new, mani_est, k, sigma_mani, K)

    # Return the final estimate of sigma:
    return perm_true(sigma, sigma_hat, n, k, perms)


################################################################################
################################################################################
################################################################################
# The following tests several algorithms after the same initializer: spectral clustering.

def initialize_and_refine(X, n, p, m, k, rho, sigma, nc, nb_size_mani, coeff_mani_est, X_new_creation_method, nb_size_X, coeff_X_new, K):
    
    A = create_adjacency_matrix(n, X, rho)
    # Do for infs and other special cases too:
    A[np.where(np.isnan(A))] = 0
    A[np.where(np.isinf(A))] = 0
    sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='kmeans')
    perms = np.array(list(itertools.permutations(range(0,k))))
    sigma_tilde_final = perm_true(sigma, sigma_tilde, n, k, perms)

    sigma_hat_nodr = refine(X, n, p, m, k, rho, sigma, sigma_tilde, perms, reduce_dimension_none, nc, estimate_manifold_avg, nb_size_mani, coeff_mani_est, create_X_new, nb_size_X, coeff_X_new, K)
    sigma_hat_dr = refine(X, n, p, m, k, rho, sigma, sigma_tilde, perms, reduce_dimension_llp, nc, estimate_manifold_avg, nb_size_mani, coeff_mani_est, create_X_new, nb_size_X, coeff_X_new, K)

    return sigma_tilde_final, sigma_hat_nodr, sigma_hat_dr


