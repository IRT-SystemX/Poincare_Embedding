import math
import torch
import numpy as np
import tqdm
import function 
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def aphi(sigma):
    return math.pow(sigma,2)+math.pow(sigma,4) + \
        ( ( ((math.pow(sigma,3))*math.sqrt(2)*math.exp(-(math.pow(sigma,2))/2)))/(math.sqrt(math.pi)*math.erf(sigma/(math.sqrt(2)))) )

sigma_pos = torch.linspace(0.01,6, 400) 
sigma_inverse = torch.Tensor([aphi(sigma_pos[i].item()) for i in range(len(sigma_pos))])
class RiemanianTools(object):
    PI_CST = pow((2*math.pi),2/3)
    SQRT_CST = math.sqrt(2)
    ERF_CST = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)

    # error function 
    @staticmethod
    def erf(x):
        return torch.sign(x)*torch.sqrt(1-torch.exp(-x*x*(4/np.pi+RiemanianTools.ERF_CST*x*x)/(1+RiemanianTools.ERF_CST*x**2)))
    # error function approximate by tanh
    @staticmethod
    def erft(x):
        return torch.tanh(x)
    @staticmethod
    def normalization_factor(sigma):
        return RiemanianTools.PI_CST * sigma * torch.exp((sigma**2)/2)*RiemanianTools.erf(sigma/(RiemanianTools.SQRT_CST))
    # gaussian probability density function
    # z = NxD
    # mu = MxD 
    # sigma = M
    @staticmethod
    def pdf(z, mu, sigma, distance):
        # z and mu dim NxMxD
        N, D, M = tuple(z.shape) +(mu.shape[0],)
        z = z.unsqueeze(1).expand(N, M, D)
        mu = mu.unsqueeze(0).expand(N, M, D)
        
        # sigma dim -> NxM
        sigma = sigma.unsqueeze(0).expand(N,M)

        # compute the exponential part
        d_to_mu = distance(z, mu)
        exp_pdf = torch.exp(-((d_to_mu**2)/(2*sigma**2)))

        # compute normalization factor
        norm_factor = RiemanianTools.normalization_factor(sigma)
        return exp_pdf/norm_factor
    
    @staticmethod
    def log(z, y):
        # qn = (y-z)/(1+z.conjugate()*y)
        return function.logMap(z, y)
    @staticmethod
    def exp(z, v):
        # v_square = v.norm(2,-1)/(1 - z.norm(2,-1)**2)
        # theta = torch.atan(v[:,0]/v[:,1])
        return function.expMap(z, v)




    @staticmethod
    def phi(values):
        index = ((sigma_inverse.unsqueeze(0).expand(values.size(0), sigma_inverse.size(0)) - values.unsqueeze(1).expand(values.size(0), sigma_inverse.size(0))))
        index = [i.nonzero()[0].item() if(len(i.nonzero()) != 0) else 6. for i in (index > 0)]

        return sigma_pos[index]


class RiemanianEM(object):
    @staticmethod
    def static_barycenter(z, lr, tau, omega_mu, max_iter=math.inf):
        N, D, M = tuple(z.shape) +(omega_mu.shape[1],)
        cvg = math.inf
        barycenter = z.mean(0)
        barycenter = barycenter.unsqueeze(0).expand(M, D)
        z = z.unsqueeze(1).expand(N, M, D)
        # SGD 
        iteration = 0
        while(cvg>tau and max_iter>iteration):
            iteration+=1

            mean_nw = RiemanianTools.log(barycenter.unsqueeze(0).expand(N, M, D), z)
            mean_w = (mean_nw * omega_mu.unsqueeze(-1).expand(N, M, D)).mean(0)

            # update weight step
            barycenter = RiemanianTools.exp(barycenter, lr * mean_w)

            cvg = torch.sqrt((mean_w.norm(2,-1)**2)/(1 - barycenter.norm(2,-1)**2)).max()
        return barycenter

    # w = M
    # pdfs = NxM
    @staticmethod
    def static_omega_mu(w, pdfs):
        N, M = tuple(pdfs.shape)
        w = w.unsqueeze(0).expand(N, M)
        w_pdf = w * pdfs
        return w_pdf/ w_pdf.sum(-1, keepdim=True).expand(N, M)

    # w = M
    # pdfs = NxM
    @staticmethod
    def static_update_w(w, pdfs):
        return RiemanianEM.static_omega_mu(w, pdfs).mean(0)

    @staticmethod
    def static_update_sigma(z, w, mu, pdfs, distance):
        omega_mu = RiemanianEM.static_omega_mu(w, pdfs)
        omega = omega_mu.unsqueeze(-1).expand(len(z), omega_mu.shape[-1], z.shape[-1])
        z = z.unsqueeze(1).expand_as(omega)
        n =  ( omega_mu * (distance(mu.unsqueeze(0).expand_as(omega), z.expand_as(omega))**2)).sum(0)
        return RiemanianTools.phi(n/omega_mu.sum(0))

    def __init__(self, n_gaussian, space_size, distance,  init_mod="rand", verbose=True):
        self._n_g = n_gaussian
        self._s_s = space_size
        self._distance = distance
        self._mu = torch.rand(n_gaussian, 2)-0.5
        self._sigma = torch.rand(n_gaussian)/10 + 0.4 
        self._w = torch.ones(n_gaussian)/n_gaussian
        self._verbose = verbose
        self._init_mod = init_mod
        self._started = False
        if(self._verbose):
            print("Initial values : ")
            print("\t mu -> ", self._mu)
            print("\t sigma -> ", self._sigma)
            print("\t weight -> ", self._w)
        

    def update_w(self, z):
        self._w = RiemanianEM.static_update_w(self._w,
                                            RiemanianTools.pdf(
                                                z, self._mu, 
                                                self._sigma, self._distance) 
                                          )
    def update_mu(self, z, lr, tau):
        self._mu = RiemanianEM.static_barycenter(z, lr, tau,
                                                    RiemanianTools.pdf(
                                                        z, self._mu, 
                                                        self._sigma, self._distance
                                                    ), max_iter=math.inf)

    def update_sigma(self, z):
        self._sigma = RiemanianEM.static_update_sigma(z, self._w, self._mu,
                                                        RiemanianTools.pdf(
                                                            z, self._mu, 
                                                            self._sigma, self._distance
                                                        ),
                                                       self._distance)

    def fit(self, z, max_iter=20, lr_mu=5e-3, tau_mu=5e-3):
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
                self.update_mu(z, lr_mu, tau_mu)
                self.update_sigma(z)
            
    def getParameters(self):
        return  self._w, self._mu, self._sigma