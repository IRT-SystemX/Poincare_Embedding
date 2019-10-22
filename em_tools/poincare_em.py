import math
import cmath
import torch
import numpy as np
import tqdm


from em_tools import poincare_kmeans as kmh
from function_tools import distribution_function as df
from function_tools import poincare_function as pf
from function_tools import poincare_alg as pa

class RiemannianEM(object):
    def __init__(self, dim, n_gaussian, init_mod="kmeans-hyperbolic", verbose=True):
        self._n_g = n_gaussian
        self._d = dim
        self._distance = pf.distance

        self._mu = (torch.rand(n_gaussian, dim) - 0.5)/dim
        self._sigma = torch.rand(n_gaussian)/10 +0.1
        self._w = torch.ones(n_gaussian)/n_gaussian

        self._verbose = verbose
        self._init_mod = init_mod
        self._started = False
        if(self._verbose):
            print("Initial values : ")
            print("\t mu -> ", self._mu)
            print("\t sigma -> ", self._sigma)
            print("\t weight -> ", self._w)
        self.zeta_phi = df.ZetaPhiStorage(torch.arange(5e-2, 1.5, 0.05), dim)

    def update_w(self, z, wik, g_index=-1):
        # get omega mu

        if(g_index > 0):
            self._w[g_index] = wik[:, g_index].mean()
        else:
            self._w = wik.mean(0)

    def update_mu(self, z, wik, lr_mu, tau_mu, g_index=-1, max_iter=50):
        N, D, M = z.shape + (wik.shape[-1],)
        # print(self._mu)
        if(g_index>0):
            self._mu[g_index] = pa.barycenter(z, wik[:, g_index], lr_mu, tau_mu, max_iter=max_iter, normed=True).squeeze()
        else:
            self._mu = pa.barycenter(z.unsqueeze(1).expand(N, M, D), wik, lr_mu,  tau_mu, max_iter=max_iter, normed=True).squeeze()
        # print("1",self._mu)
    def update_sigma(self, z, wik, g_index=-1):
        N, D, M = z.shape + (self._mu.shape[0],)
        if(g_index>0):
            dtm = ((self._distance(z, self._mu[:,g_index].expand(N))**2) * wik[:, g_index]).sum()/wik[:, g_index].sum()
            self._sigma[:, g_index] = self.phi(dtm)
        else:
            dtm = ((self._distance(z.unsqueeze(1).expand(N,M,D), self._mu.unsqueeze(0).expand(N,M,D))**2) * wik).sum(0)/wik.sum(0)
            # print("dtms ", dtm.size())
            self._sigma = self.zeta_phi.phi(dtm)        

    def _expectation(self, z):
        # computing wik 
        pdf = df.gaussianPDF(z, self._mu, self._sigma, norm_func=self.zeta_phi.zeta) 
        # print("pdf.size()->", pdf.size())
        if(pdf.mean() != pdf.mean()):
            print("EXPECTATION : pdf contain not a number elements")
            quit()
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        if(p_pdf.sum(-1).min() <= 1e-10):
            print("EXPECTATION : pdf.sum(-1) contain zero")
            p_pdf[p_pdf.sum(-1) <= 1e-4] = 1e-4
            
        wik = p_pdf/p_pdf.sum(-1, keepdim=True).expand_as(pdf)
        if(wik.mean() != wik.mean()):
            print("EXPECTATION : wik contain not a number elements")
            quit()
        # print(wik.mean(0))
        if(wik.sum(1).mean() <= 1-1e-4 and wik.sum(1).mean() >= 1+1e-4 ):
            print("EXPECTATION : wik don't sum to 1")
            print(wik.sum(1))
            quit()
        return wik

    def _maximization(self, z, wik, lr_mu=5e-3, tau_mu=1e-4, max_iter_bar=math.inf):
        self.update_w(z, wik)
        if(self._w.mean() != self._w.mean()):
            print("UPDATE : w contain not a number elements")
            quit()            
        self.update_mu(z, wik, lr_mu=lr_mu, tau_mu=tau_mu, max_iter=max_iter_bar)
        if(self._mu.mean() != self._mu.mean()):
            print("UPDATE : mu contain not a number elements")
            quit()      
        self.update_sigma(z, wik)
        if(self._sigma.mean() != self._sigma.mean()):
            print("UPDATE : sigma contain not a number elements")
            quit()  

    def fit(self, z, max_iter=5, lr_mu=5e-3, tau_mu=1e-4):
        progress_bar = tqdm.trange(max_iter) if(self._verbose) else range(max_iter)
        # if it is the first time function fit is called
        if(not self._started):
            # using kmeans for initializing means
            if(self._init_mod == "kmeans-hyperbolic"):
                if(self._verbose):
                    print("Initialize means using kmeans hyperbolic algorithm")
                km = kmh.PoincareKMeansNInit(self._n_g, n_init=20)
                km.fit(z)
                self._mu = km.cluster_centers_
            if(self._verbose):
                print("\t mu -> ", self._mu)
                print("\t sigma -> ", self._sigma)
            self._started = True
        for epoch in progress_bar:
            wik = self._expectation(z)
            self._maximization(z, wik, lr_mu=lr_mu, tau_mu=1e-4)

    def get_parameters(self):
        return  self._w, self._mu, self._sigma

    def get_pik(self, z):
        N, D, M = z.shape + (self._mu.shape[0],)
        pdf = df.gaussianPDF(z, self._mu, self._sigma, norm_func=self.zeta_phi.zeta) 
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        if(p_pdf.sum(-1).min() == 0):
            print("EXPECTATION : pdf.sum(-1) contain zero")
            #same if we set = 1
            p_pdf[p_pdf.sum(-1) == 0] = 1e-8
        wik = p_pdf/p_pdf.sum(-1, keepdim=True).expand_as(pdf)
        return wik  
    
    def predict(self, z):
        # print(self._mu)
        N, D, M = z.shape + (self._mu.shape[0],)
        pdf = df.gaussianPDF(z, self._mu, self._sigma, norm_func=self.zeta_phi.zeta) 
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        return p_pdf.max(-1)[1]
