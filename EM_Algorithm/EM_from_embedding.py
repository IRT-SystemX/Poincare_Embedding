import numpy as np
import os

from Math_Functions.Riemannian.Gaussian_PDF import *
from Math_Functions.Riemannian.Barycenter_Riemannian import *
from Visualization.Gaussian_Mixture_Visualization import *
from EM_Algorithm.Read_From_File import *
from Performance_Measures.Ground_Truth.GT_Performance_Check import *

#Z is a set of N observed samples
#iter number of iterations

def EM( iter,  M, Z):

    N = len(Z[0])                                  # Number of nodes in data

    print('Data matrix contains ', N,' nodes')

    # Barycenter approximation parameters

    tau = 0.005
    lmbd = 0.005

    weights_table = []
    variances_table = []
    barycentres_table = []

    #Generation of all the random initial values
    for dimension_index in range(len(Z)):

        weights = np.empty(M)                   #Called varomega_nu, quantifies how much a gaussian participate in the mix

        for i in range(len(weights)):
            weights[i] = 1/len(weights)

        # Average means of each gaussian
        barycentres = np.random.uniform(low = -0.5, high = 0.5, size = M)+1j*np.random.uniform(low = -0.5, high = 0.5, size = M)

        variances = np.random.uniform(low = 0.8, high = 0.9, size = M)  #Variances of each gaussian

        print('Initial values for dimension', dimension_index)

        print('\tWeights\n',weights)
        print('\tBarycentres\n', barycentres)
        print('\tVariances \n', variances)

        #Iteration until an Iter number of times while applying the update functions of the EM algorithm

        weights_old = weights.copy()
        barycentres_old = barycentres.copy()
        variances_old = variances.copy()

        for i in range(iter):

            print('Iteration ', i)

            for j in range(M):   #j is the mu index

                #print('\tGaussian number ', j)

                weights[j] = N_mu(Z[dimension_index], weights[j], weights, variances[j], variances, barycentres[j], barycentres)/N

                #print('\t\t Weights', weights)

            #for j in range(M):  # j is the mu index

                barycentres[j] = Riemannian_barycenter_weighted(Z[dimension_index], tau, lmbd, weights[j], weights, barycentres[j], barycentres,variances[j], variances)

                #print('\t\t Barycentres', barycentres)

        #for j in range(M):  # j is the mu index

                variances[j] = Variance_update2(Z[dimension_index], weights[j], weights, variances[j], variances, barycentres[j], barycentres)

                #variances[j]= Variance_update(Z,barycentres[j])

                #print('\t\t Variances', variances)

            # print('Old weights', weights_old)
            # print('Old variances', variances_old)
            # print('Old barycentres', barycentres_old)

            if( np.array_equal(weights_old, weights) and np.array_equal(barycentres_old, barycentres) and np.array_equal(variances_old,variances)):
                print('!!!!!!!!!!!!!!!!EM Convergence achieved!!!!!!!!!!!!!!!!')
                break


        weights_table.append(weights)
        variances_table.append(variances)
        barycentres_table.append(barycentres)

    print('Weights', weights_table)
    print('Barycentres', barycentres_table)
    print('Variances', variances_table)

    return weights_table, variances_table, barycentres_table



#Function to apply the EM algorithm
def EM_no_init( iter,  M, Z, weights_table, variances_table, barycentres_table):

    N = len(Z[0])                                  # Number of nodes in data

    print('Data matrix contains ', N,' nodes')

    # Barycenter approximation parameters

    tau = 0.005
    lmbd = 0.005

    #Generation of all the random initial values
    for dimension_index in range(len(Z)):

        weights = weights_table[dimension_index]                   #Called varomega_nu, quantifies how much a gaussian participate in the mix

        # Average means of each gaussian
        barycentres = barycentres_table[dimension_index]

        variances = variances_table[dimension_index]

        print('Initial values for dimension', dimension_index)

        print('\tWeights\n',weights)
        print('\tBarycentres\n', barycentres)
        print('\tVariances \n', variances)

        #Iteration until an Iter number of times while applying the update functions of the EM algorithm

        weights_old = weights.copy()
        barycentres_old = barycentres.copy()
        variances_old = variances.copy()

        for i in range(iter):

            print('Iteration ', i)

            for j in range(M):   #j is the mu index

                #print('\tGaussian number ', j)

                weights[j] = N_mu(Z[dimension_index], weights[j], weights, variances[j], variances, barycentres[j], barycentres)/N

                #print('\t\t Weights', weights)

            #for j in range(M):  # j is the mu index

                barycentres[j] = Riemannian_barycenter_weighted(Z[dimension_index], tau, lmbd, weights[j], weights, barycentres[j], barycentres,variances[j], variances)

                #print('\t\t Barycentres', barycentres)

        #for j in range(M):  # j is the mu index

                variances[j] = Variance_update2(Z[dimension_index], weights[j], weights, variances[j], variances, barycentres[j], barycentres)

                #variances[j]= Variance_update(Z,barycentres[j])

                #print('\t\t Variances', variances)

            # print('Old weights', weights_old)
            # print('Old variances', variances_old)
            # print('Old barycentres', barycentres_old)

            if( np.array_equal(weights_old, weights) and np.array_equal(barycentres_old, barycentres) and np.array_equal(variances_old,variances)):
                print('!!!!!!!!!!!!!!!!EM Convergence achieved!!!!!!!!!!!!!!!!')
                break


        weights_table.append(weights)
        variances_table.append(variances)
        barycentres_table.append(barycentres)

    print('Weights', weights_table)
    print('Barycentres', barycentres_table)
    print('Variances', variances_table)

    return weights_table, variances_table, barycentres_table