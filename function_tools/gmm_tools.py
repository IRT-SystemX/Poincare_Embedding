import numpy as np
import math
import torch

pi_2_3 = pow((2*math.pi),2/3)
a_for_erf = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)

def erf_approx(x):
    return torch.sign(x)*torch.sqrt(1-torch.exp(-x*x*(4/np.pi+a_for_erf*x*x)/(1+a_for_erf*x*x)))



def weighted_gmm_pdf(w, z, mu, sigma, distance):
    #print(w.size())
    z_u = z.unsqueeze(1).expand(z.size(0), len(mu), z.size(1))
    mu_u = mu.unsqueeze(0).expand_as(z_u)

    distance_to_mean = distance(z_u, mu_u)
    sigma_u = sigma.unsqueeze(0).expand_as(distance_to_mean)
    distribution_normal = torch.exp(-((distance_to_mean)**2)/(2 * sigma_u**2))
    zeta_sigma = pi_2_3 * sigma *  torch.exp((sigma**2/2) * erf_approx(sigma/math.sqrt(2)))
    # print("dist ", distribution_normal.size())
    return w*(distribution_normal/zeta_sigma.unsqueeze(0).expand_as(distribution_normal).detach())