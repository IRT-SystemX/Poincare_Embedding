import numpy as np
import math
import torch
from function_tools import poincare_function as pf

pi_2_3 = pow((2*math.pi),2/3)
a_for_erf = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)

def erf_approx(x):
    return torch.sign(x)*torch.sqrt(1-torch.exp(-x*x*(4/np.pi+a_for_erf*x*x)/(1+a_for_erf*x*x)))



def weighted_gmm_pdf(w, z, mu, sigma, distance, norm_factor=None):
    z_u = z.unsqueeze(1).expand(z.size(0), len(mu), z.size(1))
    mu_u = mu.unsqueeze(0).expand_as(z_u)
    distance_to_mean = distance(z_u, mu_u)
    distribution_normal = torch.exp(-((distance_to_mean)**2)/(2 * sigma**2))
    if(norm_factor is None):
        zeta_sigma = pi_2_3 * sigma.detach() *  torch.exp((sigma.detach()**2/2) * erf_approx(sigma/math.sqrt(2)))
    else:
        zeta_sigma = math.sqrt(2*math.pi) *sigma
    return (distribution_normal/zeta_sigma.unsqueeze(0).expand_as(distribution_normal).detach())


# class LogLossHyperbolicWeightedGMM(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, d_z_mu, w, sigma):
#         with torch.no_grad():
#             pdf_e = torch.exp(-((d_z_mu)**2)/(2 * sigma.expand_as(d_z_mu)**2))
#             pdf_n = pi_2_3 * sigma *  torch.exp((sigma**2/2) * erf_approx(sigma/math.sqrt(2)))
#             pdf = pdf_e/pdf_n.expand_as(pdf_e)
#             ctx.save_for_backward(d_z_mu, w, sigma.expand_as(d_z_mu))
#             return -((w * torch.log(pdf)).sum()/sigma.shape[1])
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         d_z_mu, w, sigma = ctx.saved_tensors
#         with torch.no_grad():
#             return (w * (d_z_mu/sigma)) * grad_output.expand_as(w),w.new([0]) ,w.new([0])


# def hwgloss(d_z_mu, w, sigma):
#     return LogLossHyperbolicWeightedGMM.apply(d_z_mu, w , sigma)