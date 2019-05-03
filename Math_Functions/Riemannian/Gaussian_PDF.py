import math
from Math_Functions.Riemannian.Distance_Riemannian import *


def Variance_update():
    return 0

def omega_nu(z, weight, weights, variances, barycentre, barycentres):

    nominator = weight * Gaussian_PDF(z, barycentre, variances)

    denominator = 0
    for i in range(len(weights)):
        denominator = denominator + weights[i]*Gaussian_PDF(z, barycentres[i], variances)

def N_nu (Z, weight, weights, variances, barycentre, barycentres):

    result = 0
    for i in range(len(Z)):
        result = result + omega_nu(Z[i], weight, weights, variances,barycentre, barycentres)

    return result

def Normalizing_Factor (Sigma):
    return pow((2*math.pi),2/3)*Sigma * math.exp((Sigma*Sigma)/2)*math.erf(Sigma/(math.sqrt(2)))


def Gaussian_PDF (X, Mean, Sigma):

    Z = Normalizing_Factor(Sigma)
    return (1/Z)*math.exp(-pow(Riemannian_distance(X, Mean),2))


def Gaussian_Mixture (X, Mean, Sigma, weights):

    k = len(weights)
    pdf = 0

    for i in range(k):
        pdf = pdf + weights[k]*Gaussian_PDF(X,Mean[i], Sigma[i])

    return pdf