import numpy as np
import math
import torch
from torch import nn
from function_tools import poincare_function as pf

pi_2_3 = pow((2*math.pi),2/3)
a_for_erf = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)
ZETA_CST = math.sqrt(math.pi/2)


def weighted_gmm_pdf(w, z, mu, sigma, distance, norm_func=None):
    #print(w.size())
    z_u = z.unsqueeze(1).expand(z.size(0), len(mu), z.size(1))
    mu_u = mu.unsqueeze(0).expand_as(z_u)

    distance_to_mean = distance(z_u, mu_u)
    sigma_u = sigma.unsqueeze(0).expand_as(distance_to_mean)
    distribution_normal = torch.exp(-((distance_to_mean)**2)/(2 * sigma_u**2))
    if(norm_func is None):
        zeta_sigma = pi_2_3 * sigma *  torch.exp((sigma**2/2) * erf_approx(sigma/math.sqrt(2)))
    else:
        zeta_sigma = norm_func(sigma)
    # print("dist ", distribution_normal.size())
    return w*(distribution_normal/zeta_sigma.unsqueeze(0).expand_as(distribution_normal).detach())

# TODO: validate the function
def zeta(sigma, N ,binomial_coefficient=None):      
    # we set the binomial_coefficient in parameters to avoid 
    # to compute it each time function is called
    if(N>2):
        M = sigma.shape[0]
        sigma_u = sigma.unsqueeze(0).expand(N,M)
        if(binomial_coefficient is None):
            # we compute coeficient
            v = torch.arange(N)
            v[0] = 1
            n_fact = v.prod()
            k_fact = torch.cat([v[:i].prod().unsqueeze(0) for i in range(1, v.shape[0]+1)],0)
            nmk_fact = k_fact.flip(0)
            binomial_coefficient = n_fact/(k_fact * nmk_fact)  
        binomial_coefficient = binomial_coefficient.unsqueeze(-1).expand(N,M).float()
        range_ = torch.arange(N ,device=sigma.device).unsqueeze(-1).expand(N,M).float()
        ones_ = torch.ones(N ,device=sigma.device).unsqueeze(-1).expand(N,M).float()

        alternate_neg = (-ones_)**(range_)
        ins = (((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2)
        ins_squared = ((((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2))**2
        as_o = (1+torch.erf(ins)) * torch.exp(ins_squared)
        bs_o = binomial_coefficient * as_o
        r = alternate_neg * bs_o

        return ZETA_CST * sigma * r.sum(0) * (1/(2**(N-1)))
    else:

        M = sigma.shape[0]
        sigma_u = sigma.unsqueeze(0).expand(N,M)
        if(binomial_coefficient is None):
            # we compute coeficient
            v = torch.arange(N)
            v[0] = 1
            n_fact = v.prod()
            k_fact = torch.cat([v[:i].prod().unsqueeze(0) for i in range(1, v.shape[0]+1)],0)
            nmk_fact = k_fact.flip(0)
            binomial_coefficient = n_fact/(k_fact * nmk_fact)  
        binomial_coefficient = binomial_coefficient.unsqueeze(-1).expand(N,M).float()
        range_ = torch.arange(N ,device=sigma.device).unsqueeze(-1).expand(N,M).float()
        ones_ = torch.ones(N ,device=sigma.device).unsqueeze(-1).expand(N,M).float()

        alternate_neg = (-ones_)**(range_)
        ins = (((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2)
        ins_squared = ((((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2))**2
        as_o = (1+torch.erf(ins)) * torch.exp(ins_squared)
        bs_o = binomial_coefficient * as_o
        r = alternate_neg * bs_o

        # print(">2D sigma", ZETA_CST * sigma * r.sum(0) * (1/(2**(N-1))))
        
        a = torch.exp((sigma**2)/2) * (1 + torch.erf((sigma)/math.sqrt(2)))
        b = torch.exp((-(sigma**2))/2) *(1 + torch.erf((-sigma)/math.sqrt(2)))
        # print("2D sigma", (a - b) * ZETA_CST * sigma * 0.5)
        return (a - b) * ZETA_CST * sigma * 0.5
def log_grad_zeta(x, N):
    sigma = nn.Parameter(x)
    # print(sigma.grad)
    binomial_coefficient=None
    M = sigma.shape[0]
    sigma_u = sigma.unsqueeze(0).expand(N,M)
    if(binomial_coefficient is None):
        # we compute coeficient
        v = torch.arange(N)
        v[0] = 1
        n_fact = v.prod()
        k_fact = torch.cat([v[:i].prod().unsqueeze(0) for i in range(1, v.shape[0]+1)],0)
        nmk_fact = k_fact.flip(0)
        binomial_coefficient = n_fact/(k_fact * nmk_fact)  
    binomial_coefficient = binomial_coefficient.unsqueeze(-1).expand(N,M).float()
    range_ = torch.arange(N ,device=sigma.device).unsqueeze(-1).expand(N,M).float()
    ones_ = torch.ones(N ,device=sigma.device).unsqueeze(-1).expand(N,M).float()

    alternate_neg = (-ones_)**(range_)
    ins = (((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2)
    ins_squared = ((((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2))**2
    as_o = (1+torch.erf(ins)) * torch.exp(ins_squared)
    bs_o = binomial_coefficient * as_o
    r = alternate_neg * bs_o    
    # print("bs_o.size ", ins.size())
    # print("sigma.size", sigma.size())
    # print("erf ", torch.erf(ins))
    logv = torch.log(ZETA_CST * sigma * r.sum(0) * (1/(2**(N-1))))
    # print("zeta log ",logv )
    logv.sum().backward()
    log_grad = sigma.grad
    # print("logv.grad ", sigma.grad)
    # print("ins.grad ", ins.grad)
    # print("log_grad ",log_grad)
    return sigma.grad.data

class ZetaPhiStorage(object):

    def __init__(self, sigma, N):
        self.N = N
        self.sigma = sigma
        self.m_zeta_var = (zeta(sigma, N)).detach()

        c1 = self.m_zeta_var.sum() != self.m_zeta_var.sum() 
        c2 = self.m_zeta_var.sum() == math.inf
        c3 = self.m_zeta_var.sum() == -math.inf
        if(c1 or c2 or c3):
            print("WARNING : ZetaPhiStorage , untracktable normalisation factor :")
            max_nf = len(sigma)

            limit_nf = ((self.m_zeta_var/self.m_zeta_var) * 0).nonzero()[0].item()
            self.sigma = self.sigma[0:limit_nf]
            self.m_zeta_var = self.m_zeta_var[0:limit_nf]
            if(c1):
                print("\t Nan value in processing normalisation factor")
            if(c2 or c3):
                print("\t +-inf value in processing normalisation factor")
            
            print("\t Max variance is now : ", self.sigma[-1])
            print("\t Number of possible variance is now : "+str(len(self.sigma))+"/"+str(max_nf))




        self.phi_inv_var = (self.sigma**3 * log_grad_zeta(self.sigma, N)).detach()
        
    def zeta(self, sigma):
        N, P = sigma.shape[0], self.sigma.shape[0]
        ref = self.sigma.unsqueeze(0).expand(N, P)
        val = sigma.unsqueeze(1).expand(N,P)
        values, index = torch.abs(ref - val).min(-1)
        
        return self.m_zeta_var[index]

    def phi(self, phi_val):
        N, P = phi_val.shape[0], self.sigma.shape[0]
        ref = self.phi_inv_var.unsqueeze(0).expand(N, P)
        val = phi_val.unsqueeze(1).expand(N,P)
        # print("val ", val)
        values, index = torch.abs(ref - val).min(-1)
        return self.sigma[index]
    
    def to(self, device):
        self.sigma = self.sigma.to(device)
        self.m_zeta_var = self.m_zeta_var.to(device)
        self.phi_inv_var = self.phi_inv_var.to(device)



def euclidean_norm_factor(sigma):
    print(torch.sqrt(sigma).mean())
    return ((2*math.pi) * sigma)

def gaussianPDF(x, mu, sigma, distance=pf.distance, norm_func=zeta):
    # norm_func = zeta
    # print(x.shape, mu.shape)
    N, D, M = x.shape + (mu.shape[0],)
    # print("N, M, D ->", N, M, D)
    # x <- N x M x D
    # mu <- N x M x D
    # sigma <- N x M
    x_rd = x.unsqueeze(1).expand(N, M, D)
    mu_rd = mu.unsqueeze(0).expand(N, M, D)
    sigma_rd = sigma.unsqueeze(0).expand(N, M)
    # computing numerator
    num = torch.exp(-((distance(x_rd, mu_rd)**2))/(2*(sigma_rd)**2))
    # print("num mean ",num.mean())
    den = norm_func(sigma)
    # print("den mean ",den.mean() )
    # print("sigma",num)
    # print("den ", den)
    # print("pdf max ", (num/den.unsqueeze(0).expand(N, M)).max())
    return num/den.unsqueeze(0).expand(N, M)

####################################### TESTING #####################################
def zeta_test():
    sigma = torch.arange(1e-1, 1.4, 0.02)
    N = 10
    # print(sigma)
    res = zeta(sigma, N)

    # print(res)
    r =     log_grad_zeta(sigma, N)
    # print(r)
    # print(r * sigma**3)
    # print(res)
    for i in range(1000):
        sigma_2 = torch.rand(1)
        ZPS = ZetaPhiStorage(sigma, N)
        g = ZPS.zeta(sigma_2)
        p = ZPS.phi( sigma_2**3 * log_grad_zeta(sigma_2, N))

        if((sigma_2 < 1e-1).float().sum()>1):
            print("low")
            print(sigma_2)
            print(p-sigma_2)
        else:
            try:
                assert((torch.abs(p-sigma_2)<= 0.1).float().mean().item() == 1)
            except Exception:
                print("ERROR")
                print(sigma_2)
                print(p)
                print(p-sigma_2)
                print(sigma)
                print(ZPS.phi_inv_var)
                quit()




if __name__ == "__main__":
    zeta_test()