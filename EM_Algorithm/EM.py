import numpy as np
from Math_Functions.Riemannian.Gaussian_PDF import *
from Math_Functions.Riemannian.Barycenter_Riemannian import *


#Z is a set of N observed samples
#iter number of iterations

def EM( iter, Z  ):

    M = 10                                      # Number of Gaussians used in the mix
    N = len(Z)                                  # Number of nodes in data
    weights = np.zeros(M, dtype = 'float')      #Called varomega_nu, quantifies how much a gaussian participate in the mix
    barycentres = np.zeros(M,dtype='complex')   #Average means of each gaussian
    variances = np.zeros(M, dtype = 'complex')  #Variances of each gaussian

    #Barycenter approximation parameters

    tau = 0.005
    lmbd = 0.005

    for i in range(iter):

        for j in range(M):   #j is the mu index

            weights[j] = N_mu(Z, weights[j], weights, variances[j], variances, barycentres[j], barycentres)/N

            barycentres[j] = Riemannian_barycenter_weighted(Z, tau, lmbd, weights[j], weights, barycentres[j], barycentres, variances)

            variances[j]= Variance_update()




