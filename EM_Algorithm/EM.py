import numpy as np
from Math_Functions.Riemannian.Gaussian_PDF import *
from Math_Functions.Riemannian.Barycenter_Riemannian import *
from Visualization.Gaussian_Visualization import *


#Z is a set of N observed samples
#iter number of iterations

def EM( iter, Z , M ):

    N = len(Z)                                  # Number of nodes in data

    weights = np.empty(M)                   #Called varomega_nu, quantifies how much a gaussian participate in the mix
    for i in range(len(weights)):
        weights[i] = 1/len(weights)

    barycentres = np.random.uniform(low = 0, high = 0.5, size = M)+1j*np.random.uniform(low = 0, high = 0.5, size = M)   #Average means of each gaussian
    variances = np.random.uniform(low = 1, high = 4, size = M)  #Variances of each gaussian

    print('Initial values')

    print('\tWeights\n',weights)
    print('\tBarycentres\n', barycentres)
    print('\tVariances \n', variances)


    # weights = np.zeros(M, dtype = 'float')      #Called varomega_nu, quantifies how much a gaussian participate in the mix
    # barycentres = np.zeros(M,dtype='complex')   #Average means of each gaussian
    # variances = np.zeros(M, dtype = 'float')  #Variances of each gaussian

    #Barycenter approximation parameters

    tau = 0.005
    lmbd = 0.005

    for i in range(iter):

        print('Iteration ', i)

        for j in range(M):   #j is the mu index

            print('\tGaussian number ', j)

            weights[j] = N_mu(Z, weights[j], weights, variances[j], variances, barycentres[j], barycentres)/N

            print('\t\t Weights', weights)

            barycentres[j] = Riemannian_barycenter_weighted(Z, tau, lmbd, weights[j], weights, barycentres[j], barycentres,variances[j], variances)

            print('\t\t Barycentres', barycentres)

            variances[j]= Variance_update()

            print('\t\t Variances', variances)






