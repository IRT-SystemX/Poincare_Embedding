import math
import cmath
import torch
import numpy as np
import tqdm


from em_tools import kmeans_hyperbolic as kmh
from function_tools import distribution_function as df
from function_tools import poincare_function as pf
from function_tools import poincare_alg as pa

class RiemannianEM(object):
    @staticmethod
    def _get_zeta_value(dim, n_gaussian, start=5e-2, end=1.5, step_size=1e-2):
        sigma = torch.arange(start, end, step_size)
        lgz = df.log_grad_zeta(sigma, dim)
        return sigma, lgz * sigma**3

    def __init__(self, dim, n_gaussian, init_mod="rand", verbose=True):
        self._n_g = n_gaussian
        self._d = dim
        self._distance = pf.distance

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
        self.sigma_eml, self.zeta_eml = RiemannianEM._get_zeta_value(dim, n_gaussian)

    def update_w(self, z, wik, g_index=-1):
        # get omega mu

        if(g_index > 0):
            self._w[g_index] = wik[:, g_index].mean()
        else:
            self._w = wik.mean(0)

    def update_mu(self, z, wik, lr_mu, tau_mu, g_index=-1, max_iter=50):
        N, D, M = z.shape + (wik.shape[-1],)
        if(g_index>0):
            self._mu[g_index] = pa.barycenter(z, wik[:, g_index], lr_mu, tau_mu, max_iter=max_iter).squeeze()
        else:
            self._mu = pa.barycenter(z.unsqueeze(1).expand(N, M, D), wik, lr_mu,  tau_mu, max_iter=max_iter).squeeze()
          
    def phi(self, values):
        if(values.dim() <= 0):  

            if((self.zeta_eml>=values).sum().item() < 1):
                print("Too low variance....")
                return self.sigma_eml[0]
            return self.sigma_eml[(self.zeta_eml>values).nonzero()[0][0]]

        else:
            res = []
            for i in range(values.size(0)):
                res.append(self.phi(values[i]).unsqueeze(0))
            return torch.cat(res, 0)

    def update_sigma(self, z, wik, g_index=-1):
        N, D, M = z.shape + (self._mu.shape[0],)
        if(g_index>0):
            dtm = ((self._distance(z, self._mu[:,g_index].expand(N))**2) * wik[:, g_index]).sum()/wik[:, g_index].sum()
            self.sigma[:, g_index] = self.phi(dtm)
        else:
            dtm = ((self._distance(z.unsqueeze(1).expand(N,M,D), self._mu.unsqueeze(0).expand(N,M,D))**2) * wik).sum(0)/wik.sum(0)
            print("dtms ", dtm.size())
            self.sigma = self.phi(dtm)        

    def _expectation(self, z):
        # computing wik 
        pdf = df.gaussianPDF(z, self._mu, self._sigma) 
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        wik = p_pdf/p_pdf.sum(1, keepdim=True).expand_as(pdf)
        return wik

    def _maximization(self, z, wik, lr_mu=5e-1, tau_mu=5e-3, max_iter_bar=50):
        print(self._w)
        print("qsdfjfsdjqsdfn->",wik)
        print("qsdfjfsdjqsdfn->", self._mu)
        self.update_w(z, wik)

        self.update_mu(z, wik, lr_mu=lr_mu, tau_mu=tau_mu, max_iter=max_iter_bar)
        self.update_sigma(z, wik)

    def fit(self, z, max_iter=5, lr_mu=5e-3, tau_mu=5e-3):
        progress_bar = tqdm.trange(max_iter) if(self._verbose) else range(max_iter)
        # if it is the first time function fit is called
        if(not self._started):
            # using kmeans for initializing means
            if(self._init_mod == "kmeans-hyperbolic"):
                if(self._verbose):
                    print("Initialize means using kmeans hyperbolic algorithm")
                km = kmh.PoincareKMeans(self._n_g)
                km.fit(z)
                self._mu = km.cluster_centers_
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
        pdf = df.gaussianPDF(z, self._mu, self._sigma) 
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        wik = p_pdf/p_pdf.sum(0, keepdim=True).expand_as(pdf)  
        return wik  