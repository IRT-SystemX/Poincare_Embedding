from Visualization.Gaussian_Visualization import *
from Visualization.Gaussian_Mixture_Visualization import *

Mean = complex(0.2,0.2)
Variance = 0.7

M = 3

Weights = np.empty(M)  # Called varomega_nu, quantifies how much a gaussian participate in the mix
for i in range(len(Weights)):
    Weights[i] = 1 / len(Weights)

Means = np.random.uniform(low=0, high=0.5, size=M) + 1j * np.random.uniform(low=0, high=0.5,
                                                                                  size=M)  # Average means of each gaussian
#Variances = np.random.uniform(low=0.3, high=0.9, size=M)  # Variances of each gaussian
Variances = np.array([0.4,0.4,0.4])
print('Initial values')

print('\tWeights\n', Weights)
print('\tBarycentres\n', Means)
print('\tVariances \n', Variances)

size_random = 100

Z = np.random.uniform(low = 0, high = 0.5, size = size_random)+1j*np.random.uniform(low = 0.3, high = 0.6, size = size_random)



#EM(10, Z)


#output_file_name = 'Gaussian_Curve.pdf'

#Plot_Gaussian(Mean, Variance, output_file_name)

output_file_name = 'Gaussian_Mixture.pdf'

Plot_Gaussian_Mixture(Means, Variances, Weights, output_file_name)

