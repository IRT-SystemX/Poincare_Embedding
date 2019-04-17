import numpy as np
from Math_Functions.Riemannian.Gaussian_PDF import *



def EM( iter, Z):

    M = 10
    N = len(Z)
    weights = np.zeros(M, dtype = 'float')
    barycentres = np.zeros(M,dtype='complex')
    variances = np.zeros(M, dtype = 'complex')

    for i in range(iter):

        for j in range(M):

            weights[j] = N_nu(Z, weights[j], weights, variances, barycentres)/N

            barycentres[j] =


