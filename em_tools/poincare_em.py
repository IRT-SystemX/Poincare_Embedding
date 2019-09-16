import math
import cmath
import torch
import numpy as np
import tqdm
from function_tools import numpy_function as function
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from em_tools import kmeans_hyperbolic as kmh
from function_tools.numpy_function import RiemannianFunction
from visualisation_tools import plot_tools



class RiemannianTools(object):
    # gaussian probability density function
    # z = NxD
    # mu = MxD 
    # sigma = M
    @staticmethod
    def pdf(z, mu, sigma, distance):
        # z and mu dim NxMxD
        
        N, M = z.shape + mu.shape        
        # sigma dim -> NxM
        z = np.expand_dims(z,-1).repeat(M,-1)
        mu = np.expand_dims(mu,0).repeat(N,0)
        sigma =  np.expand_dims(sigma,0).repeat(N,0)
        num = np.exp(-(distance(z, mu)**2)/(2*sigma**2))
        den = RiemannianFunction.normalization_factor(sigma)
        return num/den

    # log map similar to the one use in the previous code 
    @staticmethod
    def log(z, y):
        q = ((y-z)/(1-z.conjugate()*y))
        # a = (1 - np.abs(z) **2) * np.arctanh(np.abs(q)) * (q/np.abs(q)) 
        # b = (1 - np.abs(z) **2) * np.arctanh(np.abs(q)) * (np.cos(np.angle(q))+ np.sin(np.angle(q))*1j )
        # assert(a.sum() == b.sum()) OK
        return (1 - np.abs(z) **2) * np.arctanh(np.abs(q)) * (q/np.abs(q))

    # no change only need on one component
    @staticmethod
    def exp(z, v):
        v_square = abs(v)/(1-abs(z)*abs(z))

        theta = np.angle(v)

        numerator = (z + cmath.exp(1j * theta)) * cmath.exp(2 * v_square) + (z - cmath.exp(1j * theta))
        denominator = (1 + z.conjugate() * cmath.exp(1j * theta)) * cmath.exp(2 * v_square) + (1 - z.conjugate() * cmath.exp(1j * theta))
        result1 = numerator / denominator

        result = result1.real + result1.imag * 1j

        return result



class RiemannianEM(object):
    @staticmethod
    def static_barycenter(mu, wik, z, lr, tau, max_iter=math.inf, g_index=0, distance=RiemannianFunction.riemannian_distance):
        N, M = wik.shape
        cvg = math.inf
        barycenter = mu[g_index]

        # SGD 
        iteration = 0
        while(cvg>tau and max_iter>iteration):
            iteration+=1
            grad_value = (RiemannianTools.log(barycenter.repeat(N), z) * wik[:, g_index]).sum()/wik[:, g_index].sum()
            barycenter = RiemannianTools.exp(barycenter, lr * grad_value)
            cvg = np.sqrt((np.abs(grad_value)**2)/((1 - np.abs(barycenter)**2)**2))
        return barycenter

    # w = M
    # pdfs = NxM
    @staticmethod
    def static_omega_mu(w, pdfs):
        N, M = pdfs.shape
        w = np.expand_dims(w,0).repeat(N,0)
        w_pdf = w * pdfs
        return w_pdf/ np.expand_dims(w_pdf.sum(-1),-1).repeat(M,-1)


    @staticmethod
    def static_update_sigma(z, wik, mu,distance, g_index=0):
        N, M = wik.shape
        mu = mu[g_index].repeat(N,0)
        n = ((wik[:,g_index] * (distance(z, mu)**2) )).sum(0)
        return RiemannianFunction.phi(n/wik[:,g_index].sum(0))

    def __init__(self, n_gaussian, init_mod="rand", verbose=True):
        self._n_g = n_gaussian
        self._distance = RiemannianFunction.riemannian_distance
        self._mu = np.random.uniform(low = -0.5, high = 0.5, size = self._n_g)+1j*np.random.uniform(low = -0.5, high = 0.5, size = self._n_g)
        self._sigma = np.random.uniform(low = 0.3, high = 0.4, size = self._n_g) 
        self._w = np.ones(n_gaussian)/n_gaussian
        self._verbose = verbose
        self._init_mod = init_mod
        self._started = False
        if(self._verbose):
            print("Initial values : ")
            print("\t mu -> ", self._mu)
            print("\t sigma -> ", self._sigma)
            print("\t weight -> ", self._w)
        

    def update_w(self, wik):
        self._w = wik.mean(0)

    def update_mu(self, z, wik,  lr, tau, g_index=0):

        self._mu[g_index] = RiemannianEM.static_barycenter(self._mu, wik, z, lr, tau,
                                                    max_iter=200,
                                                    g_index=g_index
                                                )

    def update_sigma(self, z, wik,  g_index):
        self._sigma[g_index] = RiemannianEM.static_update_sigma(z,  wik, self._mu,
                                                       self._distance, g_index=g_index)

    def fit(self, z, max_iter=5, lr_mu=5e-3, tau_mu=5e-3):
        progress_bar = tqdm.trange(max_iter) if(self._verbose) else range(max_iter)
        # if it is the first time function fit is called
        z_in = z
        if(not self._started):
            # using kmeans for initializing means
            if(self._init_mod == "kmeans"):
                if(self._verbose):
                    print("Initialize means using kmeans algorithm")
                km = KMeans(self._n_g)
                km.fit(z.numpy())
                self._mu = km.cluster_centers_[:,0] +km.cluster_centers_[:,1] *1j
            if(self._init_mod == "kmeans-hyperbolic"):
                if(self._verbose):
                    print("Initialize means using kmeans hyperbolic algorithm")
                km = kmh.RiemannianKMeans(self._n_g)
                km.fit(z.numpy())
                self._mu = km.cluster_centers_[:,0] +km.cluster_centers_[:,1] *1j
<<<<<<< HEAD:em_tools/poincare_em.py
=======
                for g in range(self._n_g):
                    self.update_sigma(z, wik, g_index=g)
>>>>>>> efbcd512c2f4e561aa70a9fc64940113bc4c1f4d:em_tools/em_original.py
            if(self._verbose):
                print("Weight")
                print("\t mu -> ", self._mu)
                print("\t sigma -> ", self._sigma)
            self._started = True
        z = z[:,0].numpy()+z[:,1].numpy() *1j

        for epoch in progress_bar:
            wik = self.get_pik(z)
            self.update_w(wik)
<<<<<<< HEAD:em_tools/poincare_em.py
=======

            for g in range(self._n_g):
                
                self.update_mu(z, wik, lr_mu, tau_mu, g_index=g)
                self.update_sigma(z, wik, g_index=g)

>>>>>>> efbcd512c2f4e561aa70a9fc64940113bc4c1f4d:em_tools/em_original.py

            for g in range(self._n_g):
                
                self.update_mu(z, wik, lr_mu, tau_mu, g_index=g)
                self.update_sigma(z, wik, g_index=g)
                
    def getParameters(self):
        mu_torch = torch.cat((torch.Tensor(self._mu.real).unsqueeze(-1),torch.Tensor(self._mu.imag).unsqueeze(-1)),-1)
        return  torch.Tensor(self._w), mu_torch, torch.Tensor(self._sigma)

    def get_pik(self, z):
        return RiemannianEM.static_omega_mu(self._w,RiemannianTools.pdf(
                                                    z, self._mu, 
                                                    self._sigma, self._distance))
    def getPik(self, z):
        z = z[:,0].numpy()+z[:,1].numpy() *1j
        return torch.Tensor(RiemannianEM.static_omega_mu(self._w,RiemannianTools.pdf(
                                                    z, self._mu, 
                                                    self._sigma, self._distance)))