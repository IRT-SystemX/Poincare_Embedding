import numpy as np
from Math_Functions.Riemannian.Distance_Riemannian import *
from Math_Functions.Riemannian.Log_Riemannian import *

#Assume p ==1 or p==2
def distance_gradient(theta,a,p):
    z = complex(theta[0],theta[1])
    y = complex(a[0], a[1])

    if(p ==1):
        c = -Log_Riemannian(z,y)/(Riemannian_distance(z,y))
    else:
        #c = -p*(np.power(Riemannian_distance(z,y),(p-2)))*Log_Riemannian(z,y)
        c = -p * Log_Riemannian(z, y)
    return c



def distance_gradient_perturbation(theta,a,p):
    z = complex(theta[0],theta[1])
    y = complex(a[0], a[1])
    perturbation = 1e-5
    if(p ==1):
        c = -Log_Riemannian(z,y)/(Riemannian_distance(z,y)+perturbation)
    else:
        #c = -p*(np.power(Riemannian_distance(z,y),(p-2)))*Log_Riemannian(z,y)
        c = -p * Log_Riemannian(z, y)
    return c