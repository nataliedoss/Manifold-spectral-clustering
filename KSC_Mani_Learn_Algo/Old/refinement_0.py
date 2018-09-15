#################################################################################
#################################################################################
#################################################################################
# Code for Refinement Clustering Algorithm
# Packages
import numpy as np
import random
import sklearn.cluster
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import csv
import time
# For LASSO:
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
# For the function way of reading in the MNIST data:
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from pylab import *
from numpy import *
# For the mlxtend way of reading in the MNIST data:
import mlxtend 
from mlxtend.data import loadlocal_mnist

# Random number generator:
rng = np.random.RandomState(0)


#################################################################################
#################################################################################
#################################################################################
# Create some functions used throughout:

# Calculate the squared norm of a random vector:
def norm_squared(x):
    ret = []
    for i in x: ret.append(i**2)
    ret_final = 0
    for i in ret: ret_final = ret_final + i
    return ret_final

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
# Create the manifolds for the simulations:

# Circles:
def create_circles(n, k, m, d, sigma, b, n_turns):
    manifolds = np.zeros((n, m))
    r = np.zeros(k)
    r[0] = 1
    for i in range(0,k):
        r[i] = r[i-1] + d
    for l in range(1, k):
        t = rng.uniform(-np.pi, np.pi, len(sigma[sigma==l]))
        manifolds[sigma == l, 0] = r[l]*np.cos(t)
        manifolds[sigma == l, 1] = r[l]*np.sin(t)    
    return manifolds

# Curves:
def create_curves(n, k, m, d, sigma, b, n_turns):
    manifolds = np.zeros((n, m))
    r = np.zeros(k)
    r[0] = 0
    for i in range(1,k):
        r[i] = r[i-1] + d
    for l in range(0, k):
        t = rng.uniform(0, b, len(sigma[sigma==l]))
        manifolds[sigma == l, 0] = t
        manifolds[sigma == l, 1] = np.cos(t*2) + r[l]
    # The final curve:
    #t = rng.uniform(0, b, len(sigma[sigma==k-1]))
    #manifolds[sigma==k-1, 0] = np.sin(t*2) + (1.5*d)
    #manifolds[sigma==k-1, 1] = t 
    return manifolds

# Rainbow:
def create_rainbow(n, k, m, d, sigma, b, n_turns):
    manifolds = np.zeros((n, m))
    t2 = rng.uniform(-b, b, len(sigma[sigma==1]))
    manifolds[sigma == 0, 0] = t2 + d
    manifolds[sigma == 0, 1] = t2**2
    for l in range(1,k):
        t = rng.uniform(1, b, len(sigma[sigma==l]))
        manifolds[sigma == l, 0] = d # change if make k bigger
        manifolds[sigma == l, 1] = t
    return manifolds

# Lines: 
def create_lines(n, k, m, d, sigma, b, n_turns):
    manifolds = np.zeros((n, m))
    r = np.zeros(k) 
    r[0] = 0
    for i in range(1, k):
        r[i] = r[i-1] + d
    for l in range(k):
        t = rng.uniform(0, b, len(sigma[sigma==l]))
        manifolds[sigma == l, 0] = r[l]
        manifolds[sigma == l, 1] = t
    return manifolds

# Swiss rolls:
def create_swiss_rolls(n, k, m, d, sigma, b, n_turns):
    manifolds = np.zeros((n, m))
    max_rot = np.pi * n_turns
    r = np.zeros(k) 
    r[0] = 1
    for i in range(1, k):
        r[i] = r[i-1] + d
    for l in range(0, k):
        t = rng.uniform(0, 1, len(sigma[sigma==l])) 
        manifolds[sigma == l, 0] = r[l] * t * np.cos(t * max_rot) 
        manifolds[sigma == l, 1] = r[l] * t * np.sin(t * max_rot) 
        #manifolds[sigma == l, 2] = rng.uniform(-1, 1, (n/k))
    return manifolds

################################################################################
################################################################################
################################################################################
# Create full data with noise:

def create_noisy_data(p, epsilon, manifolds):
    mean = [0]*p
    cov = np.zeros((p,p), float)
    np.fill_diagonal(cov, epsilon)
    Z = np.random.multivariate_normal(mean, cov, n).T
    Z = Z.reshape(n,p)
    first = manifolds + Z[:,(0,m-1)]
    X = np.concatenate((first, Z[:,range(m,p)]), axis=1)
    return X

################################################################################
################################################################################
################################################################################
# Create sigma_tilde_final. We have two ways:

# Fake way:
def create_fake_sigma_tilde(A, n, k, sigma, gamma):
    sigma_tilde = np.copy(sigma)
    number_errors = int(gamma*n)
    loc_errors = np.random.randint(0,n,number_errors)
    a = np.array(range(k))
    for i in loc_errors:
        sigma_tilde[i] = np.random.choice(a[a != sigma[i]], replace=False)
    return sigma_tilde

# Via spectral clustering:
def create_adjacency_matrix(n, X, rho):
    A1 = np.zeros((n,n))
    A3 = np.zeros((n,n))
    for i in range(0,n):
        A1[i,] = norm_squared(X[i,])
        A3[i,] = norm_squared(X[i,])

    A2 = np.dot(X, X.T)
    A = A1 + A3.T - (2*A2)
    A = np.exp(-A/rho)
    np.fill_diagonal(A, 0)
    return A

def create_sc_sigma_tilde(A, n, k, sigma, gamma):
    sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, eigen_solver="arpack")
    return sigma_tilde


###############################################################################
###############################################################################
###############################################################################
# Estimate the manifolds:

def estimate_manifold(X, sigma_tilde, n, k, p, nb_size, coefficients):
    mani_est = np.zeros((n, p)) 
    for l in range(k):
        X_sub = X[sigma_tilde==l,]
        coefficients_sub = coefficients[sigma_tilde==l,]
        mani_sub = mani_est[sigma_tilde==l,]
        nbrs = NearestNeighbors(n_neighbors=int(nb_size[l]), algorithm='ball_tree').fit(X_sub)
        distances, indices = nbrs.kneighbors(X_sub)
        for i in range(0, len(X_sub)):
            mani_sub[i,] = np.array(np.average(X_sub[indices[i,],], axis=0, weights=coefficients[i,indices[i,]]))
        mani_est[sigma_tilde==l,] = mani_sub
    return mani_est

# Estimate the manifolds via PCA:
def estimate_manifols_pca(X, sigma_tilde, n, k, p, nb_size):
    mani_est = np.zeros((n,p))
    for l in range(k):
        X_sub = X[sigma_tilde==l,]
        mani_sub = mani_est[sigma_tilde==l,]
        nbrs = NearestNeighbors(n_neighbors=int(nb_size[l]), algorithm='ball_tree').fit(X_sub)
        distances, indices = nbrs.kneighbors(X_sub)
        for i in range(len(X_sub)):
            X_new = X_sub[indices[i,],]
            sample_mean = np.mean(X_new, axis=0)
            X_new_centered = X_new - sample_mean
            sample_cov = np.dot(X_new_centered.T, X_new_centered)/nb_size[l]
            U, s, V = np.linalg.svd(sample_cov)
            mani_sub[i,] = U[1,]
        mani_est[sigma_tilde==l,] = mani_sub
    return mani_est


###############################################################################
###############################################################################
###############################################################################
# Create new points for testing:

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
# Refine: There could be ties here! Fix:

def refine(data1, data2, sigma_tilde_final, K):
    D1 = np.zeros((n,n))
    D3 = np.zeros((n,n))
    for i in range(0,n):
        D1[i,] = norm_squared(data1[i,])
        D3[i,] = norm_squared(data2[i,])

    D2 = np.dot(data1, data2.T)
    distance = D1 + D3.T - (2*D2)
    sigma_hat = np.empty(n)
    for i in range(n):
        ind = np.argpartition(distance[i,], K)[:K] # gives indices
        nbs = sigma_tilde_final[ind]
        counts = np.empty(k)
        for j in range(k):
            counts[j] = np.sum(nbs == j)

        sigma_hat[i] = np.where(counts == counts.max())[0]
        
    return sigma_hat



################################################################################
################################################################################
################################################################################
# The full algorithm:

def create_simulated_data(n, k, sigma, p, m, epsilon, d, b, n_turns, create_manifolds):
    manifolds = create_manifolds(n, k, m, d, sigma, b, n_turns)
    X = create_noisy_data(p, epsilon, manifolds)
    return X


def algorithm(X, n, k, sigma, perms, p, m, rho, nb_size_manifolds, nb_size_X, K, lam, create_sigma_tilde):

    A = create_adjacency_matrix(n, X, rho)
    
    # Initialize:
    sigma_tilde = create_sigma_tilde(A, n, k, sigma, gamma)
    sigma_tilde_final = sigma_tilde

    # Supervised learning for dimension reduction:
    clf_l1_LR = LogisticRegression(C=lam, penalty='l1', tol=0.01)
    clf_l1_LR.fit(X, sigma_tilde_final)
    coef_l1 = clf_l1_LR.coef_.ravel()
    features = np.where(coef_l1 != 0)[0]
    extra = np.zeros((n, p-len(features)))
    X_lasso = np.concatenate((X[:,features], extra), axis=1)

    # Compare the various methods:
    coeff_0 = np.repeat(1, n**2).reshape(n,n)
    list_data = [X, X_lasso]
    error_array = np.empty([len(list_data), 3, 3])

    for i in range(len(list_data)):
        data = list_data[i]
        mani_est_0 = estimate_manifold(data, sigma_tilde, n, k, p, np.repeat(1,k), coeff_0)
        mani_est_1 = estimate_manifold(data, sigma_tilde, n, k, p, nb_size_manifolds, coeff_0)
        mani_est_2 = estimate_manifold(data, sigma_tilde, n, k, p, nb_size_manifolds, A)
        X_new_0 = create_X_new(data, n, p, ((1)), coeff_0)
        X_new_1 = create_X_new(data, n, p, nb_size_X, coeff_0)
        X_new_2 = create_X_new(data, n, p, nb_size_X, A)
        list_mani_est = [mani_est_0, mani_est_1, mani_est_2]
        list_X_new = [X_new_0, X_new_1, X_new_2]
        for j in range(len(list_mani_est)):
            for l in range(len(list_X_new)):
                mani_est = list_mani_est[j]
                X_new = list_X_new[l]
                sigma_hat = refine(X_new, mani_est, sigma_tilde, K)
                sigma_hat_final = perm_true(sigma, sigma_hat, n, k, perms)
                error_array[i,j,l] = np.sum(sigma_hat_final != sigma)


    error_init = np.sum(sigma_tilde_final != sigma)


    return error_init, error_array
    return sigma_hat


################################################################################
################################################################################
################################################################################
# Manifold lists:
number_outputs = 19 # length of each vector output from algorithm() function
list_manifolds = [create_circles, create_curves, create_lines]
names_manifolds = ["Circles", "Curves", "Lines"]
number_manifolds = len(list_manifolds)
# Parameters
n = 1000
k = 2
sigma = np.repeat(np.arange(0,k), [n/k]*k)
perms = np.array(list(itertools.permutations(range(0,k))))
# Rest:
p = 20
m = 2 # Manifold full (ambient + true) dimension
epsilon = .5 # Should be a decimal.
d = 2*np.sqrt(k)
b = 2
# OR: GENERALIZE THIS CODE FOR k, n:
list_nb_size_manifolds = np.array((40, 20, 10, 10, 80, 80)).reshape(number_manifolds,k)
nb_size_X = 20
rho = p/2 # For Gaussian kernel
K = 7 # For KNN in refinement
lam = .2 # For lasso
# Extras; may not use:
n_turns = 1.5 # For Swiss rolls



################################################################################
################################################################################
################################################################################
# Test:
start_time = time.time()
create_manifolds = create_circles
nb_size_manifolds = list_nb_size_manifolds[0,]
X = create_simulated_data(n, k, sigma, p, m, epsilon, d, b, n_turns, create_manifolds)
algorithm(X, n, k, sigma, perms, p, m, rho, nb_size_manifolds, nb_size_X, K, lam, create_sc_sigma_tilde)
print("--- %s seconds ---" % (time.time() - start_time))




"""

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




################################################################################
################################################################################
################################################################################
# There's an easier way to load this too!!
# MNIST. This takes about 30 minutes to run (if trying all 19 algorithms). 
data, labels = loadlocal_mnist(images_path='train-images-idx3-ubyte',
                               labels_path='train-labels-idx1-ubyte')
# Subsetting to just a few numbers:
sigma = labels[ np.logical_or(labels==0, labels==1) ]
X = data[np.logical_or(labels==0, labels==1),]
# If want to subset more:
#subset = 1000
#X = X[0:subset,]
#sigma = sigma[0:subset]

# Assign parameters:
n = len(X)
k = 2
perms = np.array(list(itertools.permutations(range(0,k))))
p = len(X[0])
rho = 5000*p # For Gaussian kernel. We need this huge for these data!
nb_size_manifolds = np.array((100, 100))
nb_size_X = 100
lam = .01

############################################################
# Run the algorithm
start_time = time.time()
errors_mnist = algorithm(X, n, k, sigma, perms, p, m, epsilon, b, n_turns, rho, nb_size_manifolds, nb_size_X, K, lam, create_sc_sigma_tilde)
print("--- %s seconds ---" % (time.time() - start_time))

############################################################
# Store results
# Formatting the data for a csv:
errors_mnist_final = np.append(errors_mnist[0], errors_mnist[1].reshape(number_outputs-1, 1))
errors_mnist_final = errors_mnist_final.reshape(number_outputs, 1)

# Create csv file of results:
colnames = ["Algorithm", "MNIST"]
with open('errors_mnist.csv', 'wb') as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(colnames)
    for row_title, data_row in zip(rownames, errors_mnist_final):
        writer.writerow([row_title] + data_row.tolist())

# Create csv file for caption in LaTeX:
with open('errors_mnist_params.csv', 'wb') as f:
    writer = csv.writer(f, delimiter=";")   
    writer.writerow( [",n = %i; k = %i; p = %i; rho = %i; K = %i; lambda = %f; Neighborhood sizes are %i and %i" %(n, k, p, rho, K, lam, nb_size_manifolds[0], nb_size_manifolds[1])] )






################################################################################
################################################################################
################################################################################
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

from pprint import pprint
pprint(list(newsgroups_train.target_names))

newsgroups_train.filenames.shape
newsgroups_train.target.shape
newsgroups_train.target[:10]

#
from sklearn.feature_extraction.text import TfidfVectorizer
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
#

from sklearn.datasets import fetch_20newsgroups_vectorized
test = fetch_20newsgroups_vectorized(subset='train')
# Not sure what we have here! Is it an array
np.array(test) # Is it what we want? Can we run algorithm on it?
# Try it:

# Assign parameters:
X = np.array(test)
n = len(X)
k = 40
p = len(X[0])
m = 1000
rho = 5000*p # For Gaussian kernel. We need this huge for these data!
nb_size_manifolds = np.repeat(2,40)
nb_size_X = 10
K = 7
lam = .01
perms = np.array(list(itertools.permutations(range(0,k))))
X
# Run algorithm
algorithm(X, n, k, sigma, perms, p, m, rho, nb_size_manifolds, nb_size_X, K, lam, create_sc_sigma_tilde)

# Don't get this datasets
# What is p??
X[0,]





"""
################################################################################
################################################################################
################################################################################
# PLOTS 


    

################################################################################
################################################################################
################################################################################
# Create a round of data, do the clustering, and plot it:
for i in range(number_manifolds):

    create_manifolds = list_manifolds[i]
    name_manifolds = names_manifolds[i]
    manifolds = create_manifolds(n, k, m, d, sigma, b, n_turns)
    X = create_noisy_data(p, manifolds)
    A = create_adjacency_matrix(n, X, rho)
    sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, eigen_solver="arpack")
    sigma_tilde_final = perm_true(sigma, sigma_tilde, n, k, perms)

    mani_est_avg = est_mani_avg(X, sigma_tilde_final, n, k, p, nb_size_manifolds[i,])
    mani_est_kernel = est_mani_kernel(X, A, sigma_tilde_final, n, k, p, rho, nb_size_manifolds[i,])
    # Plots:
    fig = plt.figure(figsize=(40,5))
    # Manifolds with Data:
    ax0 = fig.add_subplot(151, projection="3d")
    ax0.scatter(manifolds[:,0], manifolds[:,1], c="b")
    ax0.scatter(X[:,0], X[:,1], X[:,2], c="r")
    # Manifolds
    ax1 = fig.add_subplot(152, projection="3d", sharez=ax0)
    ax1.scatter(manifolds[:,0], manifolds[:,1])
    # Estimated Manifolds (average method):
    ax2 = fig.add_subplot(153, projection="3d", sharez=ax0)
    ax2.scatter(mani_est_avg[:,0], mani_est_avg[:,1], mani_est_avg[:,2])
    # Estimated Manifolds (kernel method):
    ax3 = fig.add_subplot(154, projection="3d", sharez=ax0)
    ax3.scatter(mani_est_kernel[:,0], mani_est_kernel[:,1], mani_est_kernel[:,2], c="r")
    # Estimated Manifolds (lasso method):
    ax4 = fig.add_subplot(155, projection="3d", sharez=ax0)
    ax4.scatter(mani_est_lasso[:,0], mani_est_lasso[:,1], mani_est_lasso[:,2], c="r")

    # Add a title and save it:
    plt.suptitle( "n = %i, k = %i, p = %i, epsilon = %f, d = %f, rho = %i, Neighborhood sizes are %i, %i" %(n, k, p, epsilon, d, rho, nb_size_manifolds[i,0], nb_size_manifolds[i,1]) )
    plt.show()

    fig.savefig("plot_sims_" + name_manifolds + ".pdf")


################################################################################
################################################################################
################################################################################
# MNIST - this won't work now. Fix. 
images = X.reshape(n, 28, 28)
images_est_avg = mani_est_avg.reshape(n, 28, 28)
# Plot:
# Compare these side by side. 
fig, axes = plt.subplots(nrows=1, ncols=3)
axes[0].imshow(images[sigma==1,].mean(axis=0))
axes[1].imshow(images_est_avg[sigma_tilde_final==1,].mean(axis=0))
axes[2].imshow(images_est_avg[sigma_hat_final_0==1,].mean(axis=0))
plt.show()
fig.savefig("plot_data1.pdf")
# Same for 0: 
fig, axes = plt.subplots(nrows=1, ncols=3)
axes[0].imshow(images[sigma==0,].mean(axis=0))
axes[1].imshow(images_est_avg[sigma_tilde_final==0,].mean(axis=0))
axes[2].imshow(images_est_avg[sigma_hat_final_0==0,].mean(axis=0))
plt.show()
fig.savefig("plot_data0.pdf")


# Both are false:
np.array_equal(X, mani_est_avg)
np.allclose(X, mani_est_avg)



################################################################################
################################################################################
################################################################################
# An experiment: does this work with the real manifolds?
manifolds = create_circles(n, k, m, d, sigma, b, n_turns)
X = create_noisy_data(p, manifolds)
A = create_adjacency_matrix(n, X, rho)
# Initialize:
sigma_tilde = create_sc_sigma_tilde(A, n, k, sigma, gamma)
sigma_tilde_final = perm_true(sigma, sigma_tilde, n, k, perms)
X_new_avg = create_averaged_data(X, n, p, nb_size_X)

# Fix up true manifold:
manifolds_new = create_circles(n, k, m, d, sigma_tilde_final, b, n_turns)
extra = np.zeros((n, p-m))
manifolds_final = np.concatenate((manifolds_new, extra), axis=1)
# Refine:
sigma_hat_0 = refine(X, manifolds_final, sigma_tilde_final)
sigma_hat_1 = refine(X_new_avg, manifolds_final, sigma_tilde_final)
sigma_hat_final_0 = perm_true(sigma, sigma_hat_0, n, k, perms)
sigma_hat_final_1 = perm_true(sigma, sigma_hat_1, n, k, perms)

errors = np.sum(sigma_tilde_final != sigma), np.sum(sigma_hat_final_0 != sigma), np.sum(sigma_hat_final_1 != sigma)

"""
