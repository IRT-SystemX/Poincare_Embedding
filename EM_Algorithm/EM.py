import numpy as np
from Math_Functions.Riemannian.Gaussian_PDF import *
from Math_Functions.Riemannian.Barycenter_Riemannian import *

def EM( iter, Z):

    M = 10      #Number of gaussians used in the mix
    N = len(Z)  #Number of nodes in data
    weights = np.zeros(M, dtype = 'float')  #Called varomega_nu
    barycentres = np.zeros(M,dtype='complex')
    variances = np.zeros(M, dtype = 'complex')

    #Barycenter approximation parameters

    tau = 0.005
    lmbd = 0.005

    for i in range(iter):

        for j in range(M):

            weights[j] = N_nu(Z, weights[j], weights, variances, barycentres[j], barycentres)/N

            barycentres[j] = Riemannian_barycenter_weighted(Z, tau, lmbd, weights[j], weights, barycentres[j], barycentres, variances)

            variances[j]= Variance_update()




