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
# Charts:




################################################################################
################################################################################
################################################################################
# Cifar:
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

# Parameters:
n = len(X)
k = 2
p = len(X[0])
rho = 3500000000
A = r.create_adjacency_matrix(n, X, rho)
#A[np.where(np.isnan(A))] = 0
#A = A - np.diag(np.repeat(1,n))
sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='discretize')
perms = np.array(list(itertools.permutations(range(0,k))))
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
error_init = np.sum(sigma != sigma_tilde_final)
print (error_init + 0.0)/n







################################################################################
################################################################################
################################################################################
# Olivetti faces - too perfect - can get 0 error! Really??
from sklearn import datasets
faces = datasets.fetch_olivetti_faces()
X = faces['data']
sigma = faces['target']
X = X[np.logical_or(sigma==0, sigma==1),]
sigma = sigma[np.logical_or(sigma==0, sigma==1)]

# Parameters:
n = len(X)
k = 2
p = len(X[0])
rho = 50
A = r.create_adjacency_matrix(n, X, rho)
#A[np.where(np.isnan(A))] = 0
#A = A - np.diag(np.repeat(1,n))
sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='kmeans')
perms = np.array(list(itertools.permutations(range(0,k))))
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
error_init = np.sum(sigma != sigma_tilde_final)
print (error_init + 0.0)/n




################################################################################
################################################################################
################################################################################
# Pen digits: can get error down to only 3 (.1 percent!)
data = np.genfromtxt('pendigits.tra', delimiter=',')
X = data[:,0:15]
sigma = data[:,16]
# Subset:
X = X[np.logical_or(sigma==0, sigma==1),]
sigma = sigma[np.logical_or(sigma==0, sigma==1) ]

# Parameters:
n = len(X)
k = 2
p = len(X[0])
rho = 500000
A = r.create_adjacency_matrix(n, X, rho)
#A[np.where(np.isnan(A))] = 0
#A = A - np.diag(np.repeat(1,n))
sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='kmeans')
perms = np.array(list(itertools.permutations(range(0,k))))
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
error_init = np.sum(sigma != sigma_tilde_final)
print (error_init + 0.0)/n

# Could we refine this away?





################################################################################
################################################################################
################################################################################
# Wisconsin breast cancer:
# rho = 5000 yields error rate of 4 percent or so
data = np.genfromtxt('breast-cancer-wisconsin.data.txt', delimiter=',')
X = data[:,1:9]
sigma = data[:,10]
sigma[sigma == 2] = 0
sigma[sigma == 4] = 1

# Parameters:
n = len(X)
k = 2
p = len(X[0])
rho = 5000
A = r.create_adjacency_matrix(n, X, rho)
A[np.where(np.isnan(A))] = 0
#A = A - np.diag(np.repeat(1,n))
sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='kmeans')
perms = np.array(list(itertools.permutations(range(0,k))))
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
error_init = np.sum(sigma != sigma_tilde_final)
print (error_init + 0.0)/n







################################################################################
################################################################################
################################################################################
# Image segmentation: trouble reading it in.
data = np.genfromtxt('seismic-bumps.arff.txt', delimiter=',')







################################################################################
################################################################################
################################################################################
# Image segmentation: trouble reading it in.
data = np.genfromtxt('segmentation.data.txt', delimiter=',')




################################################################################
################################################################################
################################################################################
# Spam: About 31 percent initial error. 
data = np.genfromtxt('spambase.data.txt', delimiter=',')
X = data[:,0:56]
sigma = data[:,57]

# Parameters:
n = len(X)
k = 2
p = len(X[0])
rho = 10000000
A = r.create_adjacency_matrix(n, X, rho)
D = np.sum(A, axis=1)
D = np.diag(D)
L = D - A
coeff = np.sqrt(D)
coeff = np.linalg.inv(coeff)
W = coeff.dot(L.dot(coeff))
sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='discretize')
perms = np.array(list(itertools.permutations(range(0,k))))
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
error_init = np.sum(sigma != sigma_tilde_final)
print (error_init + 0.0)/n





################################################################################
################################################################################
################################################################################
# Cell cycle: How to read it in?
np.genfromtxt('raw_cellcycle_384_17.txt',delimiter=' ',usecols=np.arange(0,1))




################################################################################
################################################################################
################################################################################
# Arabic Digits: What are the classes? This could be a good one!
data = np.genfromtxt('Train_Arabic_Digit.txt', delimiter=' ')[:,:-1]




################################################################################
################################################################################
################################################################################
# Wine dataset. Should be near perfect on this one..
data = np.loadtxt('wine.txt', delimiter=',')
X = data[:,1:14]
sigma = data[:,0]
sigma = sigma - 1

# Parameters:
n = len(X)
k = 3
p = len(X[0])
rho = 500
A = r.create_adjacency_matrix(n, X, rho)
D = np.sum(A, axis=1)
D = np.diag(D)
L = D - A
coeff = np.sqrt(D)
coeff = np.linalg.inv(coeff)
W = coeff.dot(L.dot(coeff))
sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='discretize')
perms = np.array(list(itertools.permutations(range(0,k))))
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
error_init = np.sum(sigma != sigma_tilde_final)
print (error_init + 0.0)/n 



# Choose rho:
'''
rho_vec = np.arange(600000, 900000, 1000)
error_init_vec = np.empty(len(rho_vec))

for i in range(len(rho_vec)):
    rho = rho_vec[i]
    A = r.create_adjacency_matrix(n, X, rho)
    sigma_tilde =sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='discretize')
    sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
    error_init_vec[i] = np.sum(sigma_tilde_final != sigma)

print error_init_vec
rho_vec[np.where(error_init_vec == error_init_vec.min())]
'''



################################################################################
################################################################################
################################################################################
# Heart dataset:
# Seems 24 percent is best for initializer. Use rho = 720. 
data = np.loadtxt('SPECTF.train.txt', delimiter=',')
X = data[:,1:44]
sigma = data[:,0]


# Initialize:
n = len(X)
k = 2
p = len(X[0])
rho = 720
A = r.create_adjacency_matrix(n, X, rho)
sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='kmeans')
perms = np.array(list(itertools.permutations(range(0,k))))
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
error_init = np.sum(sigma != sigma_tilde_final)
print (error_init+0.0)/n

# Choose rho:
rho_vec = np.arange(40, 100, 1)
error_init_vec = np.empty(len(rho_vec))

for i in range(len(rho_vec)):
    rho = rho_vec[i]
    A = r.create_adjacency_matrix(n, X, rho)
    sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters=k, assign_labels='discretize')
    sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
    error_init_vec[i] = (np.sum(sigma_tilde_final != sigma) + 0.0)/n

print error_init_vec
rho_vec[np.where(error_init_vec == error_init_vec.min())]



################################################################################
################################################################################
################################################################################
# Glass. Maybe take this and make it somehow general for each dataset.
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
# Assign parameters:
n = len(X)
k = 6 # FIX 
p = len(X[0])
rho = 698700 # For Gaussian kernel
nb_size_mani = np.repeat(2,k) # Generalize this; may want to change!!!
nb_size_X = 7
lam = .01
K = 7
# Will take a long time if k large:
perms = np.array(list(itertools.permutations(range(0,k))))
nc = 2
m_est = 1


# Initialize:
rho = 698700
A = r.create_adjacency_matrix(n, X, rho)
sigma_tilde = r.create_sc_sigma_tilde(A, n, k)
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
error_init = np.sum(sigma_tilde_final != sigma)

print (error_init + 0.0)/n
print n, p, k


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
A = r.create_adjacency_matrix(n, X, rho)
sigma_tilde = r.create_sc_sigma_tilde(A, n, k)
sigma_tilde_final = r.perm_true(sigma, sigma_tilde, n, k, perms)
error_init = np.sum(sigma_tilde_final != sigma)
print (error_init + 0.0)/n






'''
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
'''
