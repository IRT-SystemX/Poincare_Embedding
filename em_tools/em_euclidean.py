import math
import torch
import numpy as np
import tqdm
from function_tools import numpy_function as function
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
class EuclideanEM(object):
    @staticmethod
    def normalization_factor(sigma):
        return torch.sqrt((2 * math.pi)**2 * sigma**2)

    @staticmethod
    def pdf(z, mu, sigma):
        # z and mu dim NxMxD
        N, D, M = tuple(z.shape) +(mu.shape[0],)
        z = z.unsqueeze(1).expand(N, M, D)
        mu = mu.unsqueeze(0).expand(N, M, D)
        
        # sigma dim -> NxM
        sigma = sigma.unsqueeze(0).expand(N,M)

        # compute the exponential part
        d_to_mu = (z-mu).norm(2,-1)
        exp_pdf = torch.exp(-((d_to_mu**2)/(2*sigma**2)))

        # compute normalization factor
        norm_factor = EuclideanEM.normalization_factor(sigma)
        return exp_pdf/norm_factor
    @staticmethod
    def static_barycenter(z, omega_mu):
        N, D, M = tuple(z.shape) +(omega_mu.shape[1],)
        omega_mu = omega_mu.unsqueeze(-1).expand(N, M, D)
        z = z.unsqueeze(1).expand(N, M, D)
        return (z*omega_mu).sum(0)/omega_mu.sum(0)
    # w = M
    # pdfs = NxM
    @staticmethod
    def static_omega_mu(w, pdfs):
        N, M = tuple(pdfs.shape)
        w = w.unsqueeze(0).expand(N, M)
        w_pdf = w * pdfs
        w_pdf[w_pdf<1e-4] = 1e-3

        return w_pdf/ w_pdf.sum(-1, keepdim=True).expand(N, M)

    # w = M
    # pdfs = NxM
    @staticmethod
    def static_update_w(w, pdfs):
        # print("pdf   ->", pdfs)
        return EuclideanEM.static_omega_mu(w, pdfs).mean(0)

    @staticmethod
    def static_update_sigma(z, w, mu, pdfs):
        omega_mu = EuclideanEM.static_omega_mu(w, pdfs)
        omega = omega_mu.unsqueeze(-1).expand(len(z), omega_mu.shape[-1], z.shape[-1])
        z = z.unsqueeze(1).expand_as(omega)
        
        return ( omega_mu * (((mu.unsqueeze(0).expand_as(omega)-z.expand_as(omega))**2).sum(-1))).sum(0) / omega_mu.sum(0)

    def __init__(self, n_gaussian, space_size, distance, init_mod="rand", verbose=True):
        self._n_g = n_gaussian
        self._s_s = space_size
        self._distance = distance
        self._mu = torch.rand(n_gaussian, 2) -0.5
        self._mu = self._mu/1.8 + torch.sign(self._mu)*0.1
        self._sigma = torch.rand(n_gaussian)/10 + 0.2
        self._w = torch.ones(n_gaussian)/n_gaussian
        self._started = False
        self._verbose = verbose
        self._init_mod = init_mod
        if(self._verbose):
            print("Initial values : ")
            print("\t mu -> ", self._mu)
            print("\t sigma -> ", self._sigma)
            print("\t weight -> ", self._w)
        

    def update_w(self, z):
        self._w = EuclideanEM.static_update_w(self._w,
                                            EuclideanEM.pdf(
                                                z, self._mu, 
                                                self._sigma) 
                                          )
    def update_mu(self, z):
        self._mu = EuclideanEM.static_barycenter(z,EuclideanEM.static_omega_mu(self._w,
                                                    EuclideanEM.pdf(
                                                        z, self._mu, 
                                                        self._sigma
                                                    )))

    def update_sigma(self, z):
        self._sigma = EuclideanEM.static_update_sigma(z, self._w, self._mu,
                                                        EuclideanEM.pdf(
                                                            z, self._mu, 
                                                            self._sigma
                                                        ))

    def fit(self, z, max_iter=20):
        progress_bar = tqdm.trange(max_iter) if(self._verbose) else range(max_iter)
        if(not self._started):
            if(self._init_mod == "kmeans"):
                print("Initialize means using kmeans algorithm")
                km = KMeans(self._n_g)
                km.fit(z.numpy())
                self._mu = torch.Tensor(km.cluster_centers_)
            self._started = True
        with torch.no_grad():
            for epoch in progress_bar:
                self.update_w(z)
                self.update_mu(z)
                self.update_sigma(z)
    def getParameters(self):
        return  self._w, self._mu, self._sigma
    
class SklearnEM(object):
    def __init__(self, n_gaussian, space_size, distance, verbose=True):
        self.gm = GaussianMixture(n_components=n_gaussian, covariance_type='spherical')
    def fit(self, z):
        self.gm.fit(z.numpy())
    def getParameters(self):
        return  torch.Tensor(self.gm.weights_), torch.Tensor(self.gm.means_), 0.2*torch.Tensor(self.gm.covariances_)/torch.Tensor(self.gm.covariances_)
