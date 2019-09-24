import torch 
from function_tools import poincare_function as pf
from function_tools import distribution_function as df

class SGALoss(object):
    @staticmethod
    def O1(x, y, distance=None):
        if(distance is None):
            distance = pf.distance
        return torch.log(1e-4+torch.sigmoid(-distance(x, y)**2))

    @staticmethod
    def O2(x, y, z, distance=None):
        if(distance is None):
            distance = pf.distance
        y_reshape = y.unsqueeze(2).expand_as(z)
        return SGALoss.O1(x, y, distance=distance) + torch.log(((distance(y_reshape,z)**2).sigmoid())).sum(-1)
    
    # x = BxD
    # pi = BxM
    # mu = BxMxD
    # sigma = BxM
    @staticmethod
    def O3(x, pi, mu, sigma, zeta_f=df.zeta, distance=None):
        if(distance is None):
            distance = pf.distance
        B, M, D = (x.shape[0],) +  mu.shape
        # computing normalisation factor
        zeta_v = zeta_f(sigma)
        # computing unormalised pdf
        x_r = x.unsqueeze(1).expand(B,M,D)
        mu_r = mu.unsqueeze(0).expand(B,M,D)
        sigma_r = sigma.unsqueeze(0).expand(B, M)
        u_pdf = torch.exp(-(distance(x_r, mu_r)**2)/(2 * sigma_r**2))
        # normalize the pdf
        n_pdf = pi * torch.log((u_pdf/zeta_v))

        # return the sum over gaussian component 
        return n_pdf.sum(-1)

class SGDLoss(object):
    @staticmethod
    def O1(x, y, distance=None):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O1(x, y, distance=distance)

    @staticmethod
    def O2(x, y, z, distance=None):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O2(x, y, z, distance=distance)
    
    # x = BxD
    # pi = BxM
    # mu = BxMxD
    # sigma = BxM
    @staticmethod
    def O3(x, pi, mu, sigma, zeta_f=df.zeta, distance=None):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O3(x, pi, mu, sigma, zeta_f=zeta_f, distance=distance)

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
        return SGDSoftmaxLoss.O1 + torch.log(((-distance(y_reshape,z)**2).exp()).sum(-1))
    