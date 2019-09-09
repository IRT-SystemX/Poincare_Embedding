import math
import cmath
import torch
import numpy as np
import tqdm
from function_tools import numpy_function as function
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from function_tools.numpy_function import RiemannianFunction


class RiemannianKMeans(object):

    def __init__(self, n_clusters, init_mod="rand", verbose=True):
        self._n_c = n_clusters
        self._distance = RiemannianFunction.riemannian_distance
        self.centroids = None
    
    def _barycenter(self, x, tau=5e-3, lr=5e-3, max_iter=math.inf):
        if(len(x) == 1):
            return x[0]
        N = x.shape[0]
        cvg = math.inf
        barycenter = x.mean()

        # SGD 
        iteration = 0
        while(cvg>tau and max_iter>iteration):
            iteration+=1
            mean_w = RiemannianFunction.log(barycenter.repeat(N), x).mean()
            if(mean_w.sum() != mean_w.sum()):
                print("mean->",mean_w)
                print(x.shape, barycenter.repeat(N).shape)
                print("barycenter->",barycenter)
                print("ERROR NAN Value")
                print("iteration nb ->",iteration)
                raise NameError('Not A Number Exception')
            # update weight step
            barycenter = RiemannianFunction.exp(barycenter, lr * mean_w)
            cvg = np.sqrt((np.abs(mean_w)**2)/((1 - np.abs(barycenter)**2)**2))
        return barycenter

    def _expectation(self, centroids, x):
        N, K = x.shape[0], self.centroids.shape[0]
        centroids = np.expand_dims(centroids,0).repeat(N,0)
        x = np.expand_dims(x,-1).repeat(K,-1)
        d_c_x = self._distance(centroids, x)
        cluster_association = d_c_x.argmin(-1)
        return cluster_association

    def _maximisation(self, x, indexes):
        centroids = []
        for i in range(self._n_c):
            lx = x[indexes == i]
            if(lx.shape[0] <= 4):
                lx = np.array([x[np.random.randint(0,len(x))]])
            centroids.append(self._barycenter(lx))
        return np.array(centroids)

    def fit(self, X, max_iter=50):
        X = X[:,0] +X[:,1] *1j
        if(self.centroids == None):
            self.centroids_index = np.random.randint(0, len(X), self._n_c)

            self.centroids = X[self.centroids_index]

        
        for iteration in range(max_iter):
            self.indexes = self._expectation(self.centroids, X)
            self.centroids = self._maximisation(X, self.indexes)
        print(self.indexes)
        self.cluster_centers_ = np.array([self.centroids.real, self.centroids.imag]).transpose()
