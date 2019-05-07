import math

from Math_Functions.Riemannian.Log_Riemannian import *
from Math_Functions.Riemannian.Exp_Riemannian import *
from Math_Functions.Riemannian.Gaussian_PDF import *


def Riemannian_barycenter_weighted(Z, tau, lmbd, weight, weights, barycentre, barycentres,variance, variances):

    norm_mu = 1e6  # Also called d

    n = len(Z)

    z_chapeau = complex(0, 0)

    #Compute an initial guess of the barycenter as the euclidean mean weighted average

    for i in range(len(Z)):
        z_chapeau = z_chapeau + Z[i]*omega_mu(Z, weights[i], weights, variances[i], variances, barycentres[i], barycentres)

    #z_chapeau = z_chapeau/n             #Also called \hat{z}_new

    #print('Moyenne du cluster',z_chapeau)

    while(norm_mu >tau):

        Z_moyenne = complex(0,0)            #also called \hat{z}_old

        for j in range(0,n):

            Z_moyenne = Z_moyenne + (Log_Riemannian(z_chapeau, Z[j]))*omega_mu(Z, weight, weights, variance, variances, barycentre, barycentres)

        #Z_moyenne = Z_moyenne/n

        z_chapeau = Exp_Riemannian(z_chapeau, lmbd*Z_moyenne)

        denominator =  ( 1-abs(z_chapeau)*abs(z_chapeau) ) * (1-abs(z_chapeau)*abs(z_chapeau))

        norm_mu = math.sqrt((abs(Z_moyenne)*abs(Z_moyenne))/denominator)

    return z_chapeau

def Riemannian_barycenter(Z, tau, lmbd):

    norm_mu = 1e6           #Also called d

    n = len(Z)

    z_chapeau = complex(0,0)

    for i in Z:
        z_chapeau = z_chapeau + i

    z_chapeau = z_chapeau/n             #Also called \hat{z}_new

    #print('Moyenne du cluster',z_chapeau)

    while(norm_mu >tau):

        Z_moyenne = complex(0,0)            #also called \hat{z}_old

        for j in range(0,n):

            Z_moyenne = Z_moyenne + (Log_Riemannian(z_chapeau, Z[j]))

        Z_moyenne = Z_moyenne/n

        z_chapeau = Exp_Riemannian(z_chapeau, lmbd*Z_moyenne)

        denominator =  ( 1-abs(z_chapeau)*abs(z_chapeau) ) * (1-abs(z_chapeau)*abs(z_chapeau))

        norm_mu = math.sqrt((abs(Z_moyenne)*abs(Z_moyenne))/denominator)

    return z_chapeau