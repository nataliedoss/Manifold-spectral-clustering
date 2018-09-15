#################################################################################
#################################################################################
#################################################################################
# Code for Refinement Clustering Algorithm
# Packages
import numpy as np
import math
import random
# import sklearn.cluster
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import timeit
import numpy.linalg
from sklearn import manifold, datasets
rng = np.random.RandomState(0)

#################################################################################
#################################################################################
#################################################################################
# Functions used throughout:
def norm_squared(x):
    ret = []
    for i in x: ret.append(i**2)
    ret_final = 0
    for i in ret: ret_final = ret_final + i
    return ret_final


################################################################################
################################################################################
################################################################################
# Parameters used throughout
# Basics: n must be divisible by n. p needs to be more than 3 I think.
# Need nb size greater than n/k.
n = 10
k = 2
g = 4
grid_length = (n*g)/k
# Clusters:
sigma = np.repeat(np.arange(1,k+1), [n/k]*k)
sigma_full = np.repeat(np.arange(1,k+1), [(n*g)/k]*k)
# Rest:
p = 3
m = 2 # Manifold full (ambient + true) dimension
epsilon = .25 # Should be a decimal.
d = 5
gamma = .1
c = .1
nb_size = c*n # Put below
rho = p # For Gaussian kernel


################################################################################
################################################################################
################################################################################
# Creating the manifolds:

################################################################################
# Swiss roll:
def create_swiss_rolls(n, k, m, d, sigma, sigma_full, g, grid_length, b, n_turns):
    manifolds = np.zeros(((n*g), m))
    manifolds_sample = np.zeros((n, m))
    max_rot = n_turns * np.pi
    r = np.zeros(k)
    t = rng.uniform(0, b, grid_length)  
    r[0] = 1
    for l in range(1, k):
        r[l] = r[l-1] + d
    for l in range(1, k+1):
        manifolds[sigma_full == l, 0] = r[l-1] * t * np.cos(t * max_rot) 
        manifolds[sigma_full == l, 1] = r[l-1] * t * np.sin(t * max_rot) 
        manifolds[sigma_full == l, 2] = rng.uniform(-1, 1, grid_length)
        sample_locations = random.sample(np.arange(0, grid_length), n/k)
        manifolds_sample[sigma == l, ] = manifolds[sample_locations,]
    return (manifolds, manifolds_sample)
        

# TEST
manifolds = create_swiss_rolls(n, k, 3, d, sigma, sigma_full, g, grid_length, b=1, n_turns=1)[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(manifolds[sigma_full==1, 1], manifolds[sigma_full==1, 0], c="r")
ax.plot(manifolds[sigma_full==2, 1], manifolds[sigma_full==2, 0], c="b")
plt.show()


################################################################################
# Circles:
def create_circles(n, k, m, d, sigma, sigma_full, g, grid_length):
    manifolds = np.zeros(((n*g), m))
    manifolds_sample = np.zeros((n, m))
    t = np.linspace(-math.pi, math.pi, grid_length)
    r = np.zeros(k)
    r[0] = 1
    for i in range(1,k):
        r[i] = r[i-1] + d
    for l in range(1, k+1):
        manifolds[sigma_full == l, 0] = r[l-1]*np.cos(t)
        manifolds[sigma_full == l, 1] = r[l-1]*np.sin(t)    
        sample_locations = random.sample(np.arange(0, grid_length), n/k)
        manifolds_sample[sigma == l, ] = manifolds[sample_locations,]
    return (manifolds, manifolds_sample)


# TESTING THIS FUNCTION:
manifolds = create_circles(n, k, m, d, sigma, sigma_full, g, grid_length)[0]
#data = create_noisy_data(p, manifolds_creation[1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(manifolds[:,0], manifolds[:,1])
plt.show()



################################################################################
# Curves:
def create_curves(n, k, m, d, sigma, sigma_full, g, grid_length, b):
    manifolds = np.zeros(((n*g), m))
    manifolds_sample = np.zeros((n, m))
    t = np.linspace(0, b, grid_length)
    r = np.zeros(k)
    r[0] = 0
    for i in range(1,k):
        r[i] = r[i-1] + d
    for l in range(1, k+1):
        manifolds[sigma_full == l, 0] = t
        manifolds[sigma_full == l, 1] = np.cos(t) + r[l-1]
        sample_locations = random.sample(np.arange(0, grid_length), n/k)
        manifolds_sample[sigma == l, ] = manifolds[sample_locations,]
    return (manifolds, manifolds_sample)


# TESTING THIS FUNCTION:
manifolds = create_curves(n, k, m, d, sigma, sigma_full, g, grid_length, b=5)[0]
#data = create_noisy_data(p, manifolds_creation[1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(manifolds[:,0], manifolds[:,1])
plt.show()




################################################################################
################################################################################
################################################################################
# Create full data with noise
def create_noisy_data(p, manifolds_sample):
    mean = [0]*p
    cov = np.zeros((p,p), float)
    np.fill_diagonal(cov, epsilon)
    Z = np.random.multivariate_normal(mean, cov, n).T
    Z = Z.reshape(n,p)
    # Combine the noise and the manifold. Think of a better name.
    first = manifolds_sample + Z[:,(0,m-1)]
    X = np.concatenate((first, Z[:,range(m,p)]), axis=1)
    return X

################################################################################
################################################################################
################################################################################
# Create sigma_tilde_final.

def create_fake_sigma_tilde(n, k, gamma):
    number_errors = gamma*n
    sigma_tilde_final = np.repeat(np.arange(1,k+1), [n/k]*k)
    loc_errors = random.sample(np.arange(0, n), int(number_errors))
    a = np.array(range(1,k+1))
    for i in loc_errors:
        sigma_tilde_final[i] = np.array( random.sample( a[a != sigma[i]], 1) )
    return sigma_tilde_final

# def create_sc_sigma_tilde(n, k):



###############################################################################
###############################################################################
###############################################################################
# Manifold Estimation and creation of other data for the refinement:

# Laplacian eigenmaps: generalize the dimension here - make it a function.
u = np.zeros((n,1))
for l in range(1, k+1):
    X_sub = X[sigma_tilde_final==l,]
    ns = len(X_sub)
    D1 = np.zeros((ns,ns))
    D3 = np.zeros((ns,ns))
    for i in range(0,ns):
        D1[i,] = norm_squared(X_sub[i,])
        D3[i,] = norm_squared(X_sub[i,])

    D2 = np.dot(X_sub, X_sub.T)
    distance = D1 + D3.T - (2*D2)
    L = distance
    U, s, V = numpy.linalg.svd(L)
    u[sigma_tilde_final==l,] = U[:,1].reshape(ns, 1)

   
# Neighborhood manifold estimation:
def estimate_manifolds(X, sigma_tilde, n, p, nb_size):
    mani_est = np.zeros((n, p)) 
    for l in range(1, k+1):
        X_sub = X[sigma_tilde==l,]
        mani_sub = mani_est[sigma_tilde==l,]
        nbrs = NearestNeighbors(n_neighbors=int(nb_size), algorithm='ball_tree').fit(X_sub)
        distances, indices = nbrs.kneighbors(X_sub)
        for i in range(0, len(X_sub)):
            mani_sub[i,] = np.array(np.mean(X_sub[indices[i,],], axis=0))
            mani_est[sigma_tilde==l,] = mani_sub

    return mani_est

# New X: 
X_new = np.zeros((n,p))
nbrs = NearestNeighbors(n_neighbors=int(nb_size), algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
X[indices[i,],]
for i in range(0, n):
    X_new[i,] = np.array(np.mean(X[indices[i,],], axis=0))



###############################################################################
###############################################################################
###############################################################################
# Refinement function:
def refine(data1, data2):
    D1 = np.zeros((n,n))
    D3 = np.zeros((n,n))
    for i in range(0,n):
        D1[i,] = norm_squared(data1[i,])
        D3[i,] = norm_squared(data2[i,])

    D2 = np.dot(data1, data2.T)
    distance = D1 + D3.T - (2*D2)

    # Assign sigma_hat:
    sigma_hat = np.array([0]*n)
    for i in range(0,n):
        sigma_hat[i] = sigma_tilde_final[ np.array(np.where(distance[i,] == distance[i,].min()))[0,0] ]

    # The final result:
    return np.sum(sigma_tilde_final != sigma), np.sum(sigma_hat != sigma)
    

###############################################################################
###############################################################################
###############################################################################
# Results:




################################################################################
################################################################################
################################################################################
# Plot things in 3D:
manifolds_creation = create_circles(n, k, sigma, p, m, epsilon, d, g)
data = create_noisy_data(p, manifolds_creation[1])

# Plot the manifolds (grid):
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(manifolds_creation[0][:,0], manifolds_creation[0][:,1])
plt.show()

# Plot the noisy data:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2])
plt.show()

# Plot the estimated manifold:


