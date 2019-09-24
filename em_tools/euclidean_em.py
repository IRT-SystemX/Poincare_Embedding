import math
import cmath
import torch
import numpy as np
import tqdm
import sklearn.cluster as skc


from em_tools import kmeans_hyperbolic as kmh
from function_tools import distribution_function as df
from function_tools import euclidean_function as ef

class EuclideanEM(object):
    def norm_ff(self, sigma):
        return df.euclidean_norm_factor(sigma, self._d)

    def __init__(self, dim, n_gaussian, init_mod="rand", verbose=True):
        self._n_g = n_gaussian
        self._d = dim
        self._distance = ef.distance

        self._mu = (torch.rand(n_gaussian, dim) -0.5)/dim
        self._sigma = torch.rand(n_gaussian)/10 +0.2
        self._w = torch.ones(n_gaussian)/n_gaussian

        self._verbose = verbose
        self._init_mod = init_mod
        self._started = False
        if(self._verbose):
            print("Initial values : ")
            print("\t mu -> ", self._mu)
            print("\t sigma -> ", self._sigma)
            print("\t weight -> ", self._w)


    def update_w(self, z, wik, g_index=-1):
        # get omega mu

        if(g_index > 0):
            self._w[g_index] = wik[:, g_index].mean()
        else:
            self._w = wik.mean(0)

    def update_mu(self, z, wik, lr_mu, tau_mu, g_index=-1, max_iter=50):
        N, D, M = z.shape + (wik.shape[-1],)
        if(g_index>0):
            self._mu[g_index] =  (wik[:, g_index].unsqueeze(-1).expand(N, D) * z).sum(0)/wik[:, g_index].sum()
        else:
            self._mu = (wik.unsqueeze(-1).expand(N, M, D) * z.unsqueeze(1).expand(N, M, D)).sum(0)/wik.sum(0).unsqueeze(-1).expand(M,D)

    def update_sigma(self, z, wik, g_index=-1):
        N, D, M = z.shape + (self._mu.shape[0],)
        if(g_index>0):
            self.sigma[:, g_index] =  ((self._distance(z, self._mu[:,g_index].expand(N))**2) * wik[:, g_index]).sum()/wik[:, g_index].sum()
        else:
            self.sigma = ((self._distance(z.unsqueeze(1).expand(N,M,D), self._mu.unsqueeze(0).expand(N,M,D))**2) * wik).sum(0)/wik.sum(0)     

    def _expectation(self, z):
        # computing wik 
        pdf = df.gaussianPDF(z, self._mu, self._sigma, norm_func=self.norm_ff, distance=self._distance) 
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        wik = p_pdf/p_pdf.sum(1, keepdim=True).expand_as(pdf)
        return wik

    def _maximization(self, z, wik, lr_mu=5e-1, tau_mu=5e-3, max_iter_bar=50):
        self.update_w(z, wik)
        self.update_mu(z, wik, lr_mu=lr_mu, tau_mu=tau_mu, max_iter=max_iter_bar)
        self.update_sigma(z, wik)

    def fit(self, z, max_iter=5, lr_mu=5e-3, tau_mu=5e-3):
        progress_bar = tqdm.trange(max_iter) if(self._verbose) else range(max_iter)
        # if it is the first time function fit is called
        if(not self._started):
            # using kmeans for initializing means
            if(self._init_mod == "kmeans"):
                if(self._verbose):
                    print("Initialize means using kmeans algorithm")
                km = skc.KMeans(self._n_g)
                km.fit(z.numpy())
                self._mu = torch.Tensor(km.cluster_centers_)
            if(self._verbose):
                print("\t mu -> ", self._mu)
                print("\t sigma -> ", self._sigma)
            self._started = True
        for epoch in progress_bar:
            wik = self._expectation(z)
            self._maximization(z, wik)

    def get_parameters(self):
        return  self._w, self._mu, self._sigma

    def get_pik(self, z):
        N, D, M = z.shape + (self._mu.shape[0],)
        pdf = df.gaussianPDF(z, self._mu, self._sigma, norm_func=self.norm_ff, distance=self._distance) 
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        wik = p_pdf/p_pdf.sum(0, keepdim=True).expand_as(pdf)  
        return wik  
    

def test():
    # we take thre clusters sampled from normal
    mu, sigma = torch.rand(2) - 0.5 , torch.rand(1)/5
    x1 = torch.rand(100,2)*sigma + mu.unsqueeze(0).expand(100,2)
    mu, sigma = torch.rand(2) - 0.5 , torch.rand(1)/5
    x2 = torch.rand(100,2)*sigma + mu.unsqueeze(0).expand(100,2)
    mu, sigma = torch.rand(2) - 0.5 , torch.rand(1)/5
    x3 = torch.rand(100,2)*sigma + mu.unsqueeze(0).expand(100,2)

    X =  torch.cat((x1, x2, x3), 0)

    EM = EuclideanEM(2, 3, init_mod="kmeans")
    EM.fit(X)