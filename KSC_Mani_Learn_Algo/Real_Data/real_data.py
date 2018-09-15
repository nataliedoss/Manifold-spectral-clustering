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
sys.path.insert(0, "/Users/nataliedoss/Dropbox/Work/Projects/KSC/Code/Algorithm_Python")
import Refinement_Algorithm_Package
from Refinement_Algorithm_Package import refinement as r



################################################################################
################################################################################
################################################################################
# Wisconsin breast cancer:
# rho = 5000 yields error rate of 4 percent or so
# Get errors when try to refine!!

# Read in the data:
data = np.genfromtxt('breast-cancer-wisconsin.data.txt', delimiter=',')
X = data[:,1:9]
sigma = data[:,10]
sigma[sigma == 2] = 0
sigma[sigma == 4] = 1

# Set parameters:
n = len(X)
p = len(X[0])
m = 2
k = 6
rho = 50
nc = 5
nb_size_mani = np.repeat(1,k)
coeff_mani_est = np.repeat(1, n**2).reshape(n,n)
X_new_creation_method = r.create_X_new
nb_size_X = 1
coeff_X_new = np.repeat(1, n**2).reshape(n,n)
K = 5

# Algorithm:
start_time = time.time()
sigma_estimates = r.initialize_and_refine(X, n, p, m, k, rho, sigma, nc, nb_size_mani, coeff_mani_est, X_new_creation_method, nb_size_X, coeff_X_new, K)
print("--- %s seconds ---" % (time.time() - start_time))


print (100*np.sum(sigma_estimates[0] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[1] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[2] != sigma) + 0.0)/n





################################################################################
################################################################################
################################################################################
# Glass:

# Read in the data:
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


# Set parameters:
n = len(X)
p = len(X[0])
m = 2
k = 6
rho = 600000
nc = 2
nb_size_mani = np.repeat(1,k)
coeff_mani_est = np.repeat(1, n**2).reshape(n,n)
X_new_creation_method = r.create_X_new
nb_size_X = 20
coeff_X_new = coeff_mani_est
K = 7

# Algorithm:
start_time = time.time()
sigma_estimates = r.initialize_and_refine(X, n, p, m, k, rho, sigma, nc, nb_size_mani, coeff_mani_est, X_new_creation_method, nb_size_X, coeff_X_new, K)
print("--- %s seconds ---" % (time.time() - start_time))


print (100*np.sum(sigma_estimates[0] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[1] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[2] != sigma) + 0.0)/n




################################################################################
################################################################################
################################################################################
# Heart dataset. It seems 24 percent is best for initializer. Use rho = 720.

# Read in the data:
data = np.loadtxt('SPECTF.train.txt', delimiter=',')
X = data[:,1:44]
sigma = data[:,0]

# Set parameters:
n = len(X)
p = len(X[0])
m = 3
k = 2
rho = 720
nc = 2
nb_size_mani = np.repeat(1,k)
coeff_mani_est = np.repeat(1, n**2).reshape(n,n)
X_new_creation_method = r.create_X_new
nb_size_X = 1
coeff_X_new = coeff_mani_est
K = 5

# Algorithm:
start_time = time.time()
sigma_estimates = r.initialize_and_refine(X, n, p, m, k, rho, sigma, nc, nb_size_mani, coeff_mani_est, X_new_creation_method, nb_size_X, coeff_X_new, K)
print("--- %s seconds ---" % (time.time() - start_time))


print (100*np.sum(sigma_estimates[0] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[1] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[2] != sigma) + 0.0)/n




################################################################################
################################################################################
################################################################################
# Pen digits: got it to one error in both cases?

# Read in the data:
data = np.genfromtxt('pendigits.tra', delimiter=',')
X = data[:,0:15]
sigma = data[:,16]
# Subset:
X = X[np.logical_or(sigma==0, sigma==1),]
sigma = sigma[np.logical_or(sigma==0, sigma==1) ]

# Set parameters:
n = len(X)
p = len(X[0])
m = 5
k = 2
rho = 5000000
nc = 10
nb_size_mani = np.repeat(1,k)
coeff_mani_est = np.repeat(1, n**2).reshape(n,n)
X_new_creation_method = r.create_X_new
nb_size_X = 1
coeff_X_new = coeff_mani_est
K = 5

# Algorithm:
start_time = time.time()
sigma_estimates = r.initialize_and_refine(X, n, p, m, k, rho, sigma, nc, nb_size_mani, coeff_mani_est, X_new_creation_method, nb_size_X, coeff_X_new, K)
print("--- %s seconds ---" % (time.time() - start_time))


print (100*np.sum(sigma_estimates[0] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[1] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[2] != sigma) + 0.0)/n








################################################################################
################################################################################
################################################################################
# Cifar:

# Read in the data:
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

file = 'cifar_data_batch_1'
cifar = unpickle(file)
X = cifar['data'] # Change data type!!
sigma = np.asarray(cifar['labels'])
X_int = X[np.logical_or(sigma==0, sigma==1),]
sigma = sigma[np.logical_or(sigma==0, sigma==1)]
X = np.array(X_int, dtype='float64')

# Set parameters:
n = len(X)
p = len(X[0])
m = 100
k = 2
rho = 5000000
nc = 50
nb_size_mani = np.repeat(1,k)
coeff_mani_est = np.repeat(1, n**2).reshape(n,n)
X_new_creation_method = r.create_X_new
nb_size_X = 20
coeff_X_new = coeff_mani_est
K = 9

# Algorithm:
start_time = time.time()
sigma_estimates = r.initialize_and_refine(X, n, p, m, k, rho, sigma, nc, nb_size_mani, coeff_mani_est, X_new_creation_method, nb_size_X, coeff_X_new, K)
print("--- %s seconds ---" % (time.time() - start_time))


print (100*np.sum(sigma_estimates[0] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[1] != sigma) + 0.0)/n
print (100*np.sum(sigma_estimates[2] != sigma) + 0.0)/n


