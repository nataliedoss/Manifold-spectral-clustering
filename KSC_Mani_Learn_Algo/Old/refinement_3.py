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
# From scipy for kmeans:
import scipy
from scipy import cluster
# Other
import itertools # For permutations
import csv # To make csv's
import time # To time things
rng = np.random.RandomState(0) # Random number generator


#################################################################################
#################################################################################
#################################################################################
# The permutation function:

# Return the permutation of sigma_est that's closest to sigma
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


################################################################################
################################################################################
################################################################################
# Initialize, i.e. create sigma_tilde.

# Via spectral clustering:
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


###############################################################################
###############################################################################
###############################################################################
# Perform dimension reduction

def reduce_dimension_svd(X, sigma_tilde, n, p, k, m_est):
    X_m = np.empty((n,p))
    for l in range(k):
        X_sub = X[sigma_tilde==l,]
        U, D, V = np.linalg.svd(X_sub, full_matrices=True)
        U_m = U[:,:m_est]
        D_m = np.diag(D[:m_est])
        V_m = V[:m_est,]
        X_sub_m = U_m.dot(D_m.dot(V_m))
        X_m[sigma_tilde==l,] = X_sub_m
    return X_m

def reduce_dimension_none(X, sigma_tilde, n, p, k, nc):
    return X

# Create: reduce_dimension_lasso, reduce_dimension_pca, reduce_dimension_le

###############################################################################
###############################################################################
###############################################################################
# Estimate the manifolds. All functions return the manifold estimate and the labels.

# Estimate the manifolds via a weighted average:
def estimate_manifold_avg(X, n, p, k, sigma_tilde, nb_size, coeff, nc):
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


# Estimate the manifolds via kmeans:
def estimate_manifold_kmeans(X, n, p, k, sigma_tilde, nb_size, coeff, nc):

    # Reduce the dimension to nc via svd:
    X_nc = reduce_dimension_svd(X, sigma_tilde, n, p, k, nc)

    # Create matrices for centers and labels of centers
    centers = np.zeros((1,p))
    sigma_centers = np.zeros((1))
    
    for l in range(k):
        centers_new = scipy.cluster.vq.kmeans(X_nc[sigma_tilde==l,], nc, iter=20, thresh=1e-05)[0]
        sigma_centers_new = np.repeat(l, len(centers_new))

        centers = np.concatenate((centers, centers_new), axis=0)
        sigma_centers = np.concatenate((sigma_centers, sigma_centers_new))


    centers_final = np.delete(centers, (0), axis=0)
    sigma_centers_final = np.delete(sigma_centers, (0))

    return centers_final, sigma_centers_final
    


###############################################################################
###############################################################################
###############################################################################
# Create new points for testing (no dependence on manifolds):

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
# The full algorithm:

def refine(X, n, p, k, rho,
           sigma, sigma_tilde, perms,
           dimension_reduction_method, nc, m_est,
           manifold_estimation_method, nb_size_mani, coeff_mani_est,
           X_new_creation_method, nb_size_X, coeff_X_new,
           K):

    # Reduce dimension: 
    X_dr = dimension_reduction_method(X, sigma_tilde, n, p, k, m_est)
    
    # Estimate manifold using dimension reduced data:
    manifold_estimation = manifold_estimation_method(X_dr, n, p, k, sigma_tilde, nb_size_mani, coeff_mani_est, nc)
    mani_est = manifold_estimation[0]
    sigma_mani = manifold_estimation[1]
    
    # Create X_new: NOT using dimension reduced version of X to average right now:
    X_new = create_X_new(X, n, p, nb_size_X, coeff_X_new)
    
    # Test:
    sigma_hat = test(X_new, mani_est, k, sigma_mani, K)

    # Calculate and return error rate:
    return perm_true(sigma, sigma_hat, n, k, perms)


################################################################################
################################################################################
################################################################################
# Create some data and try the new version:
# First check kmeans
m = 10
test = np.array(range(m)).reshape(m/2,2)
#sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
kmeans = KMeans(n_clusters=k, random_state=0).fit(test)
kmeans.labels_


