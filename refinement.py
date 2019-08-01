# Refinement Clustering Algorithm Functions

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import itertools # For permutations




# Return the permutation of sigma_est that's closest to sigma:
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

    
    def __init__(self, m, k, rho, nc,
                 nb_size_mani, coeff_mani_est,
                 nb_size_X, coeff_X_new,
                 K):
        '''
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
        '''
            
        U, D, V = np.linalg.svd(X, full_matrices=True)
        U_m = U[:,:self.m]
        D_m = np.diag(D[:self.m])
        V_m = V[:self.m,]
        return U_m.dot(D_m.dot(V_m))
    

    def reduce_dimension_llp(self, X, sigma_tilde):
        '''
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
        sigma_centers must match mani_est.
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
        '''
        n = X.shape[0]
        X_dr = self.reduce_dimension_llp(X, sigma_tilde)
        mani_est = self.estimate_manifold_avg(X_dr, sigma_tilde)

        sigma_hat = self.test(X, mani_est, sigma_tilde)
        return perm_true(sigma, sigma_hat, n, self.k, perms)


    def initialize_and_refine(self, X, sigma, perms):
        '''
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


