# Manifold spectral clustering

This is a Python3 implementation of the algorithm in "Kernel spectral clustering for manifiold mixtures: improved bounds and refinement."


## External Dependencies:
[Numpy](http://numpy.org/)

[Sklearn](https://scikit-learn.org/stable/)


## Example:

```
# Modules
from refinement import * 
from create_sim_data import * 

# Data parameters
n = 5000
k = 2
p = 20
epsilon = 1.0 
sigma = np.repeat(np.arange(0,k), [n/k]*k)
perms = np.array(list(itertools.permutations(range(k))))

# Manifold generation parameters
m_create = 2  # Full manifold dimension
d = 4*np.sqrt(k) # Distance between manifolds.
b = 2 # Manifold "size."
n_turns = 1.5 # For Swiss rolls

# Algorithm parameters
rho = p/4.0
nc = 20 
m = 2 
nb_size_mani = np.repeat(1, k)
nb_size_X = 10
K = 10 
coeff_mani_est = np.repeat(1,n**2).reshape(n, n)
coeff_X_new = np.repeat(1,n**2).reshape(n, n)


# Test:
cluster = Mani_Cluster(m, k, rho, nc, nb_size_mani, coeff_mani_est,
                       nb_size_X, coeff_X_new, K)

start_time = time.time()
sigma_estimates = cluster.initialize_and_refine(X, sigma, perms)
print("--- %s seconds ---" % (time.time() - start_time))

print(np.sum(sigma_estimates[0] != sigma)) # Test the error of regular spectral clustering
print(np.sum(sigma_estimates[1] != sigma)) # The clustering error of spectral clustering + refinement


```
