################################################################################
# Create the manifolds for the simulations:

# Modules
import numpy as np
import random
rng = np.random.RandomState(0)



#################################################################################

# Circles:
def create_circles(n, k, m, d, b, n_turns, sigma):
    manifolds = np.zeros((n, m))
    r = np.zeros(k)
    r[0] = b
    for i in range(1,k):
        r[i] = r[i-1] + d
    for l in range(k):
        t = rng.uniform(-np.pi, np.pi, len(sigma[sigma==l]))
        manifolds[sigma == l, 0] = r[l]*np.cos(t)
        manifolds[sigma == l, 1] = r[l]*np.sin(t)    
    return manifolds

# Curves:
def create_curves(n, k, m, d, b, n_turns, sigma):
    manifolds = np.zeros((n, m))
    r = np.zeros(k)
    r[0] = 0
    for i in range(1,k):
        r[i] = r[i-1] + d
    for l in range(0, k):
        t = rng.uniform(0, b, len(sigma[sigma==l]))
        manifolds[sigma == l, 0] = t
        manifolds[sigma == l, 1] = np.cos(t*2) + r[l]
    return manifolds

# Rainbow:
def create_rainbow(n, k, m, d, b, n_turns, sigma):
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
def create_lines(n, k, m, d, b, n_turns, sigma):
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
def create_swiss_rolls(n, k, m, d, b, n_turns, sigma):
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
    return manifolds



################################################################################
################################################################################
################################################################################
# Create full data with noise:

def create_noisy_data(n, p, epsilon, m, manifolds):
    mean = [0]*p
    cov = np.zeros((p,p), float)
    np.fill_diagonal(cov, epsilon)
    Z = np.random.multivariate_normal(mean, cov, n).T
    Z = Z.reshape(n,p)
    first = manifolds + Z[:,(0,m-1)]
    X = np.concatenate((first, Z[:,range(m,p)]), axis=1)
    return X

def create_simulated_data(n, p, k, epsilon, m, d, b, n_turns, sigma, create_manifolds):
    manifolds = create_manifolds(n, k, m, d, b, n_turns, sigma)
    X = create_noisy_data(n, p, epsilon, m, manifolds)
    return X
