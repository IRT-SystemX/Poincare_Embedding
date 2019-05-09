import cmath
import math
from Math_Functions.Riemannian.Distance_Riemannian import *


def Variance_update():

    return 1

def omega_mu(z, weight, weights, variance, variances, barycentre, barycentres):

    nominator = weight * Gaussian_PDF(z, barycentre, variance)

    denominator = 0
    for i in range(len(weights)):
        denominator = denominator + weights[i]*Gaussian_PDF(z, barycentres[i], variances[i])

    print('\t\tOmega_mu', nominator/denominator)
    return nominator/denominator

def N_mu (Z, weight, weights, variance, variances, barycentre, barycentres):

    result = 0
    for i in range(len(Z)):
        result = result + omega_mu(Z[i], weight, weights, variance, variances,barycentre, barycentres)

    return result

def Normalizing_Factor (Sigma):
    return pow((2*math.pi),2/3)*Sigma * math.exp((Sigma*Sigma)/2)*math.erf(Sigma/(math.sqrt(2)))


def Gaussian_PDF (X, Mean, Sigma):

    Z = Normalizing_Factor(Sigma)

    print('Normalizing factor of gaussien', Z)


    result = (1/Z)*math.exp(-pow(Riemannian_distance(X, Mean),2))

    print('Gaussiaan probability', result)
    #return Riemannian_distance(X, Mean)
    return result



def Gaussian_Mixture (X, Mean, Sigma, weights):

    k = len(weights)
    pdf = 0

    for i in range(k):
        pdf = pdf + weights[k]*Gaussian_PDF(X,Mean[i], Sigma[i])

    return pdf