import math
import cmath
import torch
import numpy as np
import tqdm
import sklearn.cluster as skc
import sklearn.mixture as skm

from function_tools import distribution_function as df
from function_tools import euclidean_function as ef

class GaussianMixtureSKLearn(skm.GaussianMixture):
    def __init__(self, n_gaussian, init_mod="rand", verbose=True):
        super(GaussianMixtureSKLearn, self).__init__(n_components=n_gaussian)  

    def fit(self, X, Y=None):
        if(Y is not None):
            super(GaussianMixtureSKLearn, self).fit(X.numpy(), Y.numpy())
        else:
            super(GaussianMixtureSKLearn, self).fit(X.numpy())
        self._w = torch.Tensor(self.weights_)
        self._mu = torch.Tensor(self.means_)
        self._sigma = torch.Tensor(self.covariances_)

    def probs(self, z):
        y = self.predict_proba(z.numpy())
        return torch.Tensor(y)

    def predict(self, z):
        return torch.Tensor(super(GaussianMixtureSKLearn, self).predict(z.numpy()))

class EuclideanEM(object):
    def norm_ff(self, sigma):
        return df.euclidean_norm_factor(sigma, self._d)

    def __init__(self, n_gaussian, init_mod="rand", verbose=True):
        self._n_g = n_gaussian

        self._distance = ef.distance

        self._verbose = verbose
        self._init_mod = init_mod
        self._started = False

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
            self._sigma[:, g_index] =  ((self._distance(z, self._mu[:,g_index].expand(N))**2) * wik[:, g_index]).sum()/wik[:, g_index].sum()
        else:
            self._sigma = ((self._distance(z.unsqueeze(1).expand(N,M,D), self._mu.unsqueeze(0).expand(N,M,D))**2) * wik).sum(0)/wik.sum(0)     

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

    def fit(self, z, max_iter=5, lr_mu=5e-3, tau_mu=5e-3, Y=None):
        progress_bar = tqdm.trange(max_iter) if(self._verbose) else range(max_iter)
        # if it is the first time function fit is called
        if(not self._started):
            self._d = z.size(-1)
            self._mu = (torch.rand(self._n_g, self._d) -0.5)/self._d
            self._sigma = torch.rand(self._n_g)/10 +0.2
            self._w = torch.ones(self._n_g)/self._n_g
        if(Y is not None):
            wik = Y.float()
            self._maximization(z, wik, lr_mu=lr_mu, tau_mu=1e-5)
            return
        else:
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
        print(pdf.mean())
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        if(p_pdf.sum(-1).min() == 0):
            print("EXPECTATION : pdf.sum(-1) contain zero")
            #same if we set = 1
            p_pdf[p_pdf.sum(-1) == 0] = 1e-8
        wik = p_pdf/p_pdf.sum(-1, keepdim=True).expand_as(pdf)
        return wik 


    def probs(self, z):
        return self.get_pik(z)


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