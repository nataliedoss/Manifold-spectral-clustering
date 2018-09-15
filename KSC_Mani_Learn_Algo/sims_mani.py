################################################################################
################################################################################
################################################################################
# Test whether regular manifold clustering (no noise) does worse when b grows


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
sys.path.insert(0, "/Users/nataliedoss/Dropbox/Work/Research/Manifold_Learning/Code/Algorithm_Python")
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
m = 2
d = 2
b = 10
n_turns = 1.5 # For Swiss rolls
sigma = np.repeat(np.arange(0,k), [n/k]*k)
perms = np.array(list(itertools.permutations(range(0,k))))


# Create data and plot
mani = cr.create_circles(n, k, m, d, b, n_turns, sigma)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(mani[:,0], mani[:,1])
plt.show()




################################################################################
################################################################################
################################################################################
# Testing various d, which in this case corresponds to manifold size
d = 5
blist = np.arange(.5,10,.5)
errors_km = np.repeat(-1, len(blist))
errors_sc = np.repeat(-1, len(blist))
for i in range(len(blist)):
    b = blist[i]
    rho = b**2
    mani = cr.create_circles(n, k, m, d, b, n_turns, sigma)
    A = r.create_adjacency_matrix(n, mani, rho)

    sigma_km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1).fit(mani).labels_  
    sigma_sc = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='kmeans')

    sigma_km_final = r.perm_true(sigma, sigma_km, n, k, perms)
    sigma_sc_final = r.perm_true(sigma, sigma_sc, n, k, perms)

    errors_km[i] = np.sum(sigma_km_final != sigma)
    errors_sc[i] = np.sum(sigma_sc_final != sigma)
    
print "d is %i" %d
print blist
print errors_km
print errors_sc

