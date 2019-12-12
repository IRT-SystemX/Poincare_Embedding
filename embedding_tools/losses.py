import torch 

from function_tools import poincare_function as pf
from function_tools import distribution_function as df
from torch.nn import functional as tf

class SGALoss(object):
    @staticmethod
    def O1(x, y, distance=None, coef=1.):
        if(distance is None):
            distance = pf.distance
        return tf.logsigmoid(-torch.clamp(((distance(x, y) * coef)**2), max=100))

    @staticmethod
    def O2(x, y, z, distance=None, coef=1.):
        if(distance is None):
            distance = pf.distance
        x_reshape = x.unsqueeze(-2).expand_as(z)
        return SGALoss.O1(x, y, distance=distance, coef=coef) + tf.logsigmoid((distance(x_reshape,z)*coef)**2).sum(-1)
    
    # x = BxD
    # pi = BxM
    # mu = BxMxD
    # sigma = BxM
    @staticmethod
    def O3(x, pi, mu, sigma, normalisation_factor, distance=None):
        if(distance is None):
            distance = pf.distance
        B, M, D = (x.shape[0],) +  mu.shape
        # print("B,M, D", B, M, D)
        # computing normalisation factor
        # print(normalisation_factor.size())
        zeta_v = normalisation_factor.unsqueeze(0).expand(B, M)
        # computing unormalised pdf
        x_r = x.unsqueeze(1).expand(B,M,D)
        mu_r = mu.unsqueeze(0).expand(B,M,D)
        # print("mu_r size ", mu_r.size())
        # print("x_r size ", x_r.size())
        sigma_r = sigma.unsqueeze(0).expand(B, M)
        u_pdf = torch.exp(-(distance(x_r, mu_r)**2)/(2 * sigma_r**2)).clamp(min=1e-10) 
        # normalize the pdf
        # print("u_pdf", (u_pdf / zeta_v).min(-1))
        # print("pi_size ", pi.min())
        # print("u_pdf size ", u_pdf.size())
        # print("zeta_v.size ", zeta_v.size())
        n_pdf = pi.squeeze() * torch.log((u_pdf/zeta_v))
        # print("n_pdf min : ", n_pdf.min())
        # print("n_pdf max : ", n_pdf.max())
        # print("n_pdf size : ", n_pdf.size())
        # print("n_pdf * pi min : ", (n_pdf * pi).min())
        # print("u_pdf log", n_pdf.sum(-1))
        # return the sum over gaussian component 
        return n_pdf.sum(-1)

# class SGDLossFast(object):
#     class O1Autograd(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, x):
#             with torch.no_grad():
#                 u_prime = torch.exp(x.clamp_max(16.0))
#                 u = torch.log(1 + u_prime)
#                 ctx.save_for_backward(u, u_prime)
#                 return u
#         @staticmethod
#         def backward(ctx, grad_output):
#             with torch.no_grad():
#                 u, u_prime = ctx.saved_tensors
#                 return (u_prime/u) * grad_output

#     class O2Autograd(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, x, y):
#             with torch.no_grad():
#                 # clamping value 
#                 u_prime = torch.exp(x.clamp_max(16.0))
#                 u = torch.log(1 + u_prime)

#                 v_prime = torch.exp(-(y.clamp_max(16.0)))
#                 v = torch.log(1 + v_prime)
#                 ctx.save_for_backward(u, u_prime, v, v_prime)
#                 # print("xclm -> ",x.clamp_max(64.0).max())
#                 # print("uv_min-> ", (u + v.sum(-1)).min())
#                 # print("uv_max-> ", (u + v.sum(-1)).max())
#                 return u + v.sum(-1)
#         @staticmethod
#         def backward(ctx, grad_output):
#             with torch.no_grad():
#                 u, u_prime, v, v_prime = ctx.saved_tensors
#                 v_grad = (-v_prime/v )
#                 if((u == u).float().mean() != 1):
#                     print("ERROR NAN VALUE IN u")
#                 if((v == v).float().mean() != 1):
#                     print("ERROR NAN VALUE IN v")
#                     quit()
#                 if(u.min() == 0):
#                     print("ERROR ZERO VALUE IN u")
#                     quit()
#                 # print("vmin -> ",v.min())
#                 # print("umin -> ",u.min())
#                 # print("vax -> ",v.max())
#                 # print("umax -> ",u.max())
#                 # print("v gradmax -> ",v_grad.max())
#                 # print("v gradmin -> ",v_grad.min())
#                 if(v.min() == 0):
#                     print("ERROR ZERO VALUE IN v")

#                 # if examples are same 
#                 if(v.min() == 0):
#                     v_grad[v < 1e-3] = 0
#                 return  (u_prime/u)  * grad_output, v_grad * grad_output.unsqueeze(-1).expand_as(v)


#     @staticmethod
#     def O1(x, y, distance=None):
#         if(distance is None):
#             distance = pf.poincare_distance_squared
#         return SGDLossFast.O1Autograd.apply(distance(x, y))

#     @staticmethod
#     def O2(x, y, z, distance=None):
#         if(distance is None):
#             distance = pf.poincare_distance_squared
#         # print("x_norm -> ", x.norm(2,-1).max())
#         # print("y_norm -> ", y.norm(2,-1).max())
#         # print("z_norm -> ", z.norm(2,-1).max())
#         # print("z_max -> ")
#         return SGDLossFast.O2Autograd.apply(distance(x, y), distance(y.unsqueeze(2).expand_as(z), z))


class SGDLoss(object):
    @staticmethod
    def O1(x, y, distance=None, coef=1.):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O1(x, y, distance=distance, coef=coef)

    @staticmethod
    def O2(x, y, z, distance=None, coef=1.):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O2(x, y, z, distance=distance, coef=coef)
    
    # x = BxD
    # pi = BxM
    # mu = BxMxD
    # sigma = BxM
    @staticmethod
    def O3(x, pi, mu, sigma, normalisation_factor, distance=None):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O3(x, pi.detach(), mu.detach(), sigma.detach(), normalisation_factor.detach(), distance=distance)

class SGDSoftmaxLoss(object):
    @staticmethod
    def O1(x, y, distance=None):
        if(distance is None):
            distance = pf.distance
        return distance(x, y)**2
    
    @staticmethod
    def O2(x, y, z, distance=None):
        if(distance is None):
            distance = pf.distance
        y_reshape = y.unsqueeze(2).expand_as(z)
        return SGDSoftmaxLoss.O1(x, y) + ( torch.log((-distance(y_reshape,z)**2).exp() )).sum(-1)
    