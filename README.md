# Manifold spectral clustering

This is a Python3 implementation of the algorithm in Kernel Spectral Clustering: Improved Bounds and Refinement.


## Example:

```
cluster = Mani_Cluster(m, k, rho, nc, nb_size_mani, coeff_mani_est,
                       nb_size_X, coeff_X_new, K)

start_time = time.time()
sigma_estimates = cluster.initialize_and_refine(X, sigma, perms)
print("--- %s seconds ---" % (time.time() - start_time))

print(np.sum(sigma_estimates[0] != sigma))
print(np.sum(sigma_estimates[1] != sigma))

```
