from Visualization.Gaussian_Visualization import *
from Visualization.Gaussian_Mixture_Visualization import *
from EM_Algorithm.EM import *


M = 4                   # Number of Gaussians used in the mix

# Weights = np.empty(M)   # Called varomega_nu, quantifies the weight of a Gaussian in the mix
#
#
# for i in range(len(Weights)):
#     Weights[i] = 1 / len(Weights)       #Initially we assume equal weights

# Average means of each gaussian
#Means = np.random.uniform(low=-0.5, high=0.5, size=M) + 1j * np.random.uniform(low=-0.5, high=0.5, size=M)
#Means = np.array([-0.8-0.2j, 0.7+0.2j, 0.1+0.1j])


#Variances = np.random.uniform(low=0.6, high=1.2, size=M)  # Variances of each gaussian
#Variances = np.array([0.8,0.8,0.6])

# print('Initial values')
# print('\tWeights\n\t\t', Weights)
# print('\tBarycentres\n')
# for i in Means:
#     print('\t\t',i)
# print('\tVariances \n\t\t', Variances)

size_random = 100

#Z = np.random.uniform(low = 0, high = 0.5, size = size_random)+1j*np.random.uniform(low = 0.3, high = 0.6, size = size_random)

filename = 'Karate\Karate_0.pkl'

EM(10,M, filename, )
#EM(10, Z)


#output_file_name = 'Gaussian_Curve.pdf'

#Plot_Gaussian(Means[0], Variances[0], output_file_name)

output_file_name = 'Gaussian_Mixture.pdf'

#Plot_Gaussian_Mixture(Means, Variances, Weights, output_file_name)

