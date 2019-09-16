import torch 
from function_tools import poincare_function as pf
from function_tools import distribution_function as df

class SGALoss(object):
    @staticmethod
    def O1(x, y):
        return torch.log(torch.sigmoid(-pf.distance(x, y)**2))

    @staticmethod
    def O2(x, y, z):
        y_reshape = y.unsqueeze(2).expand_as(z)
        return SGALoss.O1(x, y) + torch.log(((pf.distance(y_reshape,z)**2).sigmoid())).sum(-1)
    
    # x = BxD
    # pi = BxM
    # mu = BxMxD
    # sigma = BxM
    @staticmethod
    def O3(x, pi, mu, sigma, zeta_f=df.zeta):
        B, M, D = mu.shape
        # computing normalisation factor
        zeta_v = zeta_f(sigma)
        # computing unormalised pdf
        u_pdf = torch.exp(-(pf.distance(x.unsqueeze(1).expand(B,M,D), mu)**2)/(2 * sigma**2))
        # normalize the pdf
        n_pdf = pi * torch.log((u_pdf/zeta_v))

        # return the sum over gaussian component 
        return n_pdf.sum(-1)

class SGDLoss(object):
    @staticmethod
    def O1(x, y):
        return -SGALoss.O1(x, y)

    @staticmethod
    def O2(x, y, z):
        return -SGALoss.O2(x, y, z)
    
    # x = BxD
    # pi = BxM
    # mu = BxMxD
    # sigma = BxM
    @staticmethod
    def O3(x, pi, mu, sigma, zeta_f=df.zeta):
        return -SGALoss.O3(x, pi, mu, sigma, zeta_f=zeta_f)