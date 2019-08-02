# Refinement Clustering Algorithm 

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import itertools # For permutations


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


class Mani_Cluster:
    '''
    class for manifold clustering algorithm
    '''

    
    def __init__(self, m, k, rho, nc,
                 nb_size_mani, coeff_mani_est,
                 nb_size_X, coeff_X_new,
                 K):
        '''
        Args:
        m: Int. Estimated manifold dimension.
        k: Int. Estimated number of clusters.
        rho: Float. Bandwidth used in forming Gaussian kernel in adjacency matrix.
        nc: Int. Number of clusters used in local linear projection.
        nb_size_mani: Int. Neighborhood size used in nearest neighbor manifold estimation. 
        coeff_mani_est: Array(float, n x n). Used to form weighted average in nearest neighbor manifold estimation. 
        nb_size_X: Int. Neighborhood size used to form new, local-average versions of data X. 
        coeff_X_new: Array(float, n x n). Used to form weighted average if we choose to form new, local-averages versions of data X. 
        K: Int. Number of neighbors to use in refinement testing.
        '''
        
        self.m = m
        self.k = k
        self.rho = rho
        self.nc = nc
        self.nb_size_mani = nb_size_mani
        self.coeff_mani_est = coeff_mani_est
        self.nb_size_X = nb_size_X
        self.coeff_X_new = coeff_X_new
        self.K = K


    def create_adjacency_matrix(self, X):
        '''
        Args:
        X: Array(float, n x p). Signal (manifold) + noise data used to create kernel adjacency matrix.

        Returns:
        Array(float, n x n). Kernel adjacency matrix; kernel is the Gaussian kernel.
        '''
            
        n = X.shape[0]
        A1 = np.zeros((n,n))
        A3 = np.zeros((n,n))
        for i in range(n):
            A1[i,] = (np.linalg.norm(X[i,]))**2
            A3[i,] = (np.linalg.norm(X[i,]))**2

        A2 = np.dot(X, X.T)
        A = A1 + A3.T - (2*A2)
        A = np.exp(-A/self.rho)
        return A


    def lra(self, X):
        '''
        Method to compute m-rank approximation of X.

        Args:
        X: Array(float, n_sub x p). n_sub is size of dataset we wish to do linear projection on.

        Returns:
        Array(float, b_sub x p). Low rank approximation of X. 
        '''
            
        U, D, V = np.linalg.svd(X, full_matrices=True)
        U_m = U[:,:self.m]
        D_m = np.diag(D[:self.m])
        V_m = V[:self.m,]
        return U_m.dot(D_m.dot(V_m))
    

    def reduce_dimension_llp(self, X, sigma_tilde):
        '''
        Method to locally project data (project subsets of data) to nearest linear subspace of R^p.

        Args:
        X: Array(float, n x p). Signal (manifold) + noise data.
        sigma_tilde: Array(int, n x 1). 

        Returns:
        Array(float, n x p). Dimension-reduced points.
        '''
        
        n = X.shape[0]
        p = X.shape[1]
        all_manifolds_est = np.zeros((n,p))
        for l in range(self.k):
            X_mani = X[sigma_tilde == l,]
            labels_nc = KMeans(n_clusters=self.nc, init='k-means++', n_init=10, max_iter=300,
                               tol=0.0001, precompute_distances='auto', verbose=0, random_state=None,
                               copy_x=True, n_jobs=1).fit(X_mani).labels_
            if self.m < np.bincount(labels_nc).min():
                m = self.m
            else:
                m = np.bincount(labels_nc).min()
            one_manifold_est = np.zeros((len(X_mani), p))
            for j in range(self.nc):
                X_local = X_mani[labels_nc == j,]
                mani_est_local = self.lra(X_local)
                one_manifold_est[labels_nc == j,] = mani_est_local
            all_manifolds_est[sigma_tilde == l, ] = one_manifold_est

        return all_manifolds_est


    def estimate_manifold_avg(self, X, sigma_tilde):
        '''
        Method to estimate manifold using locally projected data and local averaging.

        Args:
        X: Array(float, n x p). Dimension reduced points.

        Returns:
        Array(float, n x p). Estimated manifold points.
        '''
        
        n = X.shape[0]
        p = X.shape[1]
        mani_est = np.zeros((n, p)) 
        for l in range(self.k):
            X_sub = X[sigma_tilde == l,]
            coeff_sub = self.coeff_mani_est[sigma_tilde == l,]
            mani_sub = mani_est[sigma_tilde == l,]
            nbrs = NearestNeighbors(n_neighbors=int(self.nb_size_mani[l]), algorithm='ball_tree').fit(X_sub)
            distances, indices = nbrs.kneighbors(X_sub)
            for i in range(len(X_sub)):
                mani_sub[i,] = np.array(np.average(X_sub[indices[i,],], axis = 0, weights = coeff_sub[i,indices[i,]]))
            mani_est[sigma_tilde == l,] = mani_sub

        return mani_est


    def create_X_new(self, X):
        '''
        Method to create new versions of full (not dimension reduced) data.

        Args:
        X: Array(float, n x p). Signal (manifold) + noise data.

        Returns:
        Array(float, n x p). Locally averaged versions of X. 
        '''
        
        n = X.shape[0]
        p = X.shape[1]
        X_new = np.zeros((n,p))
        nbrs = NearestNeighbors(n_neighbors=int(self.nb_size_X), algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        for i in range(n):
            X_new[i,] = np.array(np.average(X[indices[i,],], axis=0, weights=self.coeff_X_new[i,indices[i,]]))
        return X_new


    def test(self, X, mani_est, sigma_centers):
        '''
        Method to assign cluster label based on nearest manifold point.
        Sigma_centers must match mani_est.

        Args:
        X: Array(float, n x p). Signal (manifold) + noise data. 
        mani_est: Array(float, n x p). Estimated manifold points. 
        sigma_centers: Array(int, n x 1). Cluster assignments of manifold points. 

        Returns:
        Array(int, n x 1). Estimated cluster assignment.
        '''
        
        n = X.shape[0]
        nc_full = mani_est.shape[0]
        distance = np.zeros((n, nc_full)).reshape(n, nc_full)
        for i in range(n):
            for j in range(nc_full):
                distance[i,j] = np.linalg.norm(X[i,] - mani_est[j,])

        sigma_hat = np.empty(n)

        for i in range(n):
            ind = np.argpartition(distance[i,], self.K)[:self.K]
            nbs = sigma_centers[ind]
            counts = np.empty(self.k)
            for j in range(self.k):
                counts[j] = np.sum(nbs == j)
            sigma_hat[i] = np.where(counts == counts.max())[0][0] # If tie, take first one.

        return sigma_hat

        
    def refine(self, X, sigma, sigma_tilde, perms):
        '''
        Method to estimate manifold and perform testing to obtain improved estimate of cluster assignment.
        Requires the true sigma in order to obtain 
        
        Args:
        X: Array(float, n x p). Signal (manifold) + noise data. 
        sigma: Array(int, n x 1). True cluster assignment. 
        sigma_tilde: Array(int, n x 1). Initial cluster assignment (e.g., after spectral clustering). 
        perms: Array(int, (k!) x k). All permutations of sigma. 
        
        Returns:
        Array(int, n x 1). Estimated cluster assignment after refinement (best permutation). 
        '''
        
        n = X.shape[0]
        X_dr = self.reduce_dimension_llp(X, sigma_tilde)
        mani_est = self.estimate_manifold_avg(X_dr, sigma_tilde)

        sigma_hat = self.test(X, mani_est, sigma_tilde)
        return perm_true(sigma, sigma_hat, n, self.k, perms)


    def initialize_and_refine(self, X, sigma, perms):
        '''
        Args:
        X: Array(float, n x p). Signal (manifold) + noise data.
        sigma: Array(int, n x 1). True cluster assignment. 
        perms: Array(int, (k!) x k). All permutations of sigma. 

        Returns:
        sigma_tilde_final: Array(int, n x 1). Estimated cluster assignment after initialization via spectral clustering.
        sigma_hat_dr: Array(int, n x 1). Estimated cluster assignment after refinement algorithm. 
        '''
        
        n = X.shape[0]
        A = self.create_adjacency_matrix(X)
        A[np.where(np.isnan(A))] = 0
        A[np.where(np.isinf(A))] = 0
        sigma_tilde = sklearn.cluster.spectral_clustering(A, n_clusters = self.k,
                                                          assign_labels = 'kmeans')
        perms = np.array(list(itertools.permutations(range(self.k))))
        sigma_tilde_final = perm_true(sigma, sigma_tilde, n, self.k, perms)
        sigma_hat_dr = self.refine(X, sigma, sigma_tilde, perms)

        return sigma_tilde_final, sigma_hat_dr


