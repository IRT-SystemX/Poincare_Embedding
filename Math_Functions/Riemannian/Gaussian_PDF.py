import math
from Math_Functions.Riemannian.Distance_Riemannian import *

def Z_D(Sigma):

    return pow((2*math.pi),2/3)*Sigma * math.exp((Sigma*Sigma)/2)*math.erf(Sigma/(math.sqrt(2)))


def Normalizing_Factor (Sigma, n):

    C = 1

    product = 1
    for j in range(1,n-1):
        product = Z_D(Sigma/(math.sqrt(n-j)))

    Z = C * math.sqrt(2*math.pi/n)*Sigma * product

    return Z


def Gaussian_PDF (X, Mean, Sigma, n):

    Z = Normalizing_Factor(Sigma,n)
    return (1/Z)*math.exp(-pow(Riemannian_distance(X, Mean),2))


def Gaussian_Mixture (X, Mean, Sigma, n, weights):

    k = len(weights)
    pdf = 0

    for i in range(k):
        pdf = pdf + weights[k]*Gaussian_PDF(X,Mean[k], Sigma[k],n)

    return pdf