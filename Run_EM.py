from Visualization.Gaussian_Visualization import *
from Visualization.Gaussian_Mixture_Visualization import *
from EM_Algorithm.EM import *


M = 4           # Number of Gaussians used in the mix
iter_max = 10   # Maximum number of EM iterations

#size_random = 100
#Z = np.random.uniform(low = 0, high = 0.5, size = size_random)+1j*np.random.uniform(low = 0.3, high = 0.6, size = size_random)

filename = 'Karate/Karate_0.pkl'

EM( iter_max, M, filename)





