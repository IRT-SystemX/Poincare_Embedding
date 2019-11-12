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
        self.n_gaussian = n_gaussian
        self._distance = ef.distance
        super(GaussianMixtureSKLearn, self).__init__(n_components=n_gaussian, covariance_type="spherical")  

    def norm_ff(self, sigma):
        return df.euclidean_norm_factor(sigma)

    def fit(self, X, Y=None):

        if(Y is not None):
            N, M, D = X.size(0), Y.size(-1), X.size(-1)
            Y = Y.float()/(Y.float().sum(-1, keepdim=True).expand_as(Y))
            self.weights_ = Y.mean(0).numpy()
            print("w -> ",self.weights_)

            means = ((X.unsqueeze(1).expand(N,M,D) * Y.unsqueeze(-1).expand(N,M,D)).sum(0)/Y.sum(0).unsqueeze(-1).expand(M,D))
            self.means_2 = means.numpy()
            print("mean -> ", self.means_2.shape)
            print("mean[0] -> ", self.means_2[:,0])
            self.N_k = Y.sum(0)
            XmM = X.unsqueeze(1).expand(N,M,D)-means.unsqueeze(0).expand(N,M,D)
            print((XmM * XmM).sum(-1))
            self.covariances_2 = (((XmM * XmM).sum(-1)) * Y).sum(0)/Y.sum(0)/30
            print((XmM * XmM).sum(-1).mean())
            print("cov ", self.covariances_2)
            super(GaussianMixtureSKLearn, self).__init__(n_components=self.n_gaussian, covariance_type="spherical", precisions_init=(1/self.covariances_2).numpy(),weights_init=self.weights_/self.weights_.sum(), means_init=self.means_2, max_iter=50)  
            super(GaussianMixtureSKLearn, self).fit(X.numpy())
        else:
            super(GaussianMixtureSKLearn, self).fit(X.numpy())
        self._w = torch.Tensor(self.weights_)
        self._mu = torch.Tensor(self.means_2)
        self._sigma = torch.Tensor(self.covariances_)
        # self._sigma = self.covariances_2/X.size(-1)
        print("sigma ", self._sigma)
        print("w ", self._w)
        print("mean[0] ", self.means_[:,0])

    def get_pik(self, z):
        # print('sigma  ->  ', self._sigma)
        # print('sigma sqrt ->  ', torch.sqrt(self._sigma))
        
        # N, D, M = z.shape + (self._mu.shape[0],)
        # pdf = df.gaussianPDF(z, self._mu, torch.sqrt(self._sigma), norm_func=self.norm_ff, distance=self._distance) 

        # print("pdf mean", pdf[20])
        # p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        # print("ppd",p_pdf.size())
        # if(p_pdf.sum(-1).min() == 0):
        #     print("EXPECTATION : pdf.sum(-1) contain zero -> ",(p_pdf.sum(-1) == 0).sum().item())
        #     #same if we set = 1
        #     p_pdf[p_pdf.sum(-1) == 0] = 1e-8
        # wik = p_pdf/p_pdf.sum(-1, keepdim=True).expand_as(pdf)
        # print("wik 1", wik.mean(1))
        # print("wik 2", wik.mean(0))
        # wik[torch.arange(len(wik)), wik.max(1)[1]] = 1
        # wik = wik.long().float()
        # print("wik 2", wik.mean(0))
        print(torch.Tensor(super(GaussianMixtureSKLearn, self).predict_proba(z.numpy()))[0])
        return torch.Tensor(super(GaussianMixtureSKLearn, self).predict_proba(z.numpy()))


    def probs(self, z):
        return self.get_pik(z)
    def predict(self, z):
        return torch.Tensor(super(GaussianMixtureSKLearn, self).predict(z.numpy()))

class EuclideanEM(object):
    def norm_ff(self, sigma):
        return df.euclidean_norm_factor(sigma)

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
            print("sqdqsdf", self._mu.size())

    def update_sigma(self, z, wik, g_index=-1):
        N, D, M = z.shape + (self._mu.shape[0],)

        N_k = wik.sum(0)
        if(g_index>0):
            self._sigma[:, g_index] =  ((self._distance(z, self._mu[:,g_index].expand(N))**2) * wik[:, g_index]).sum()/wik[:, g_index].sum()
        else:
            self._sigma =((self._distance(z.unsqueeze(1).expand(N,M,D), self._mu.unsqueeze(0).expand(N,M,D)))**2 * wik).sum(0)/wik.sum(0)
            # ((((X.unsqueeze(1).expand(N,M,D)-means.unsqueeze(0).expand(N,M,D))**2).sum(-1)) * Y).sum(0)/Y.sum(0)
    def _expectation(self, z):
        # N, M, D = z.size(0), self._mu.size(0), z.size(-1)

        # z = z.unsqueeze(1).expand(N, M, D)
        # mu = self._mu.unsqueeze(0).expand(N,)
        # ZmM = 
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
            print("Y size ", Y.size())
            wik = Y.float()/(Y.float().sum(-1, keepdim=True).expand_as(Y))
            print("wik size ", wik.size())
            print("wik", wik[0])
            self._maximization(z, wik, lr_mu=lr_mu, tau_mu=1e-5)
            print("_sig ", self._sigma)
            print("_w", self._w)
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

        print("pdf mean", pdf[20])
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        print("ppd",p_pdf.size())
        if(p_pdf.sum(-1).min() == 0):
            print("EXPECTATION : pdf.sum(-1) contain zero -> ",(p_pdf.sum(-1) == 0).sum())
            #same if we set = 1
            p_pdf[p_pdf.sum(-1) == 0] = 1e-8
        wik = p_pdf/p_pdf.sum(-1, keepdim=True).expand_as(pdf)
        # print("wik 1", wik.mean(1))
        # print("wik 2", wik.mean(0))
        # wik[torch.arange(len(wik)), wik.max(1)[1]] = 1
        # wik = wik.long().float()
        # print("wik 2", wik.mean(0))
        return wik


    def probs(self, z):
        # by default log probs since probs can be easily untracktable
        N, M, D = z.size(0), self._mu.size(0), z.size(-1)

        z = z.unsqueeze(1).expand(N, M, D)
        mu = self._mu.unsqueeze(0).expand(N,M,D)

        ZmM = ((mu - z)**2).sum(-1)
        ZmMmS = -(1/2)  * ZmM * 1/self._sigma.unsqueeze(0).expand(N,M)
        


        nor = -(z.size(-1)/2) * (math.log(2 * math.pi) + torch.log(self._sigma))
        nor = nor.unsqueeze(0).expand(N,M)

        log_pdf = nor + ZmMmS

        log_prob = torch.log(self._w.unsqueeze(0).expand(N,M)) + log_pdf 
        print("log prob ", log_prob[0])
        return log_prob



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