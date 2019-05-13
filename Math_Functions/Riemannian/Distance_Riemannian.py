import numpy as np



def Riemannian_distance(z1,z0):

    d = float(0)

    num = 1+abs((z1-z0)/(1-z0.conjugate()*z1))

    denom = 1-abs( (z1-z0)/(1-z0.conjugate()*z1))

    d = (1/2)*np.log(num/denom)

    return d


def Riemannian_distance_perturbation(z1,z0):

    perturbation = 1e-5

    d = float(0)

    num = 1+abs((z1-z0)/(1-z0.conjugate()*z1+perturbation))

    denom = 1-abs( (z1-z0)/(1-z0.conjugate()*z1+ perturbation)) + perturbation

    d = (1/2)*np.log(num/denom)

    return d