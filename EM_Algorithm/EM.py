import numpy as np
import os

from Math_Functions.Riemannian.Gaussian_PDF import *
from Math_Functions.Riemannian.Barycenter_Riemannian import *
from Visualization.Gaussian_Mixture_Visualization import *
from EM_Algorithm.Read_From_File import *
from Performance_Measures.Ground_Truth.GT_Performance_Check import *

#Z is a set of N observed samples
#iter number of iterations

def EM( iter,  M, example_name, filename, truth_check = False):

    #Read the input graph matrix from the file placed in the Input folder

    B = Get_matrix(example_name+'/'+filename)

    Z = 1j * B[:, 1]
    Z += B[:, 0]

    N = len(Z)                                  # Number of nodes in data

    print('Data matrix contains ', N,' nodes')

    #Generation of all the random initial values

    weights = np.empty(M)                   #Called varomega_nu, quantifies how much a gaussian participate in the mix
    for i in range(len(weights)):
        weights[i] = 1/len(weights)

    # Average means of each gaussian
    barycentres = np.random.uniform(low = -0.5, high = 0.5, size = M)+1j*np.random.uniform(low = -0.5, high = 0.5, size = M)

    variances = np.random.uniform(low = 0.8, high = 0.9, size = M)  #Variances of each gaussian

    print('Initial values')

    print('\tWeights\n',weights)
    print('\tBarycentres\n', barycentres)
    print('\tVariances \n', variances)

    #Barycenter approximation parameters

    tau = 0.005
    lmbd = 0.005

    #Iteration until an Iter number of times while applying the update functionso f the EM algorithm

    for i in range(iter):

        print('Iteration ', i)

        for j in range(M):   #j is the mu index

            print('\tGaussian number ', j)

            weights[j] = N_mu(Z, weights[j], weights, variances[j], variances, barycentres[j], barycentres)/N

            print('\t\t Weights', weights)

        #for j in range(M):  # j is the mu index

            barycentres[j] = Riemannian_barycenter_weighted(Z, tau, lmbd, weights[j], weights, barycentres[j], barycentres,variances[j], variances)

            print('\t\t Barycentres', barycentres)

        #for j in range(M):  # j is the mu index

            variances[j] = Variance_update2(Z, weights[j], weights, variances[j], variances, barycentres[j], barycentres)

            #variances[j]= Variance_update(Z,barycentres[j])

            print('\t\t Variances', variances)



    #Find the class for each data node by applying the criterion

    labels = np.empty(N)

    for i in range(N):

        probabilities = np.zeros(M)

        for j in range(M):

            probabilities[j] = weights[j] * Gaussian_PDF(Z[i], barycentres[j], variances[j])

        labels[i] = np.argmax(probabilities)

    print('labels',labels)


    #Ground Truth Check
    if(truth_check == True):

        if(M>6):
            performance = Truth_Check_Large_K(example_name,labels,M)
        else:

            performance = Truth_Check_Small_K(example_name,labels, M)


        print('Performance', performance)




    #Save everything and plot

    #Creation of the output directory if it does not yet exists

    output_directory = 'Output/'+example_name+'/'+filename+'_'+str(iter)+'_iter_'+str(M)+'_M'
    try:
        os.makedirs(output_directory)
        print("Directory ", output_directory, " Created")

    except FileExistsError:
        print("Directory ", output_directory, " already exists")


    #Save to Python Pickle
    print('Saving data...')
    with open(output_directory + '/Output_Data_' + example_name + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([B, weights, barycentres, variances, performance], f)

    print('Data Saved to ', output_directory + '/Output_Data_' +example_name+ '.pkl')

    #Save to text file

    print('Saving Variances and Truth percentage to text file...')
    file = open(output_directory + '/Performances_'+example_name +'.txt', 'w')


    # file.write('Variances:\n')
    # for i in Variances:
    #     file.write(str(i) + '\n')
    file.write('Truth percentage:\n')
    file.write(str(performance))

    file.close()



    #Plotting the results

    Plot_Gaussian_Mixture(Z, barycentres, variances, weights, output_directory)






