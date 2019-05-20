import cmath
import math
import matplotlib.pyplot as plt
from Math_Functions.Riemannian.Distance_Riemannian import *



def inverse_phi( Sigma):

    return math.pow(Sigma,2)+math.pow(Sigma,4)+ ( ( ((math.pow(Sigma,3))*math.sqrt(2)*math.exp(-(math.pow(Sigma,2))/2)))/(math.sqrt(math.pi)*math.erf(Sigma/(math.sqrt(2)))) )

def phi (argument_phi):

    print('Argument of Phi', argument_phi)

    result = 0

    Number_Steps = 200

    Sigma = np.linspace(0.01, 2, Number_Steps)

    inverse_phi_table = np.zeros(Number_Steps)

    for i in range(len(inverse_phi_table)):
        inverse_phi_table[i] = inverse_phi(Sigma[i])


    look_up_point = -1

    for i in range(len(inverse_phi_table)):

        if(inverse_phi_table[i]>argument_phi):
            look_up_point = i
            break


    if(look_up_point == -1):
        print('Error: Could not estimate Sigma, Please increase Phi inverse function range')


    result = Sigma[look_up_point]


    # plt.plot(Sigma, inverse_phi_table)
    #
    #
    # plt.plot(argument_phi, Sigma[look_up_point],'ro')
    # plt.show()

    return result


def phi_2 (Z, Mean):

    argument_phi = 0

    for i in range(len(Z)):

        argument_phi = argument_phi + pow(Riemannian_distance(Z[i], Mean),2)

    argument_phi = argument_phi/len(Z)

    print('Argument of Phi', argument_phi)

    result = 0

    Number_Steps = 200

    Sigma = np.linspace(0.01, 2, Number_Steps)

    inverse_phi_table = np.zeros(Number_Steps)

    for i in range(len(inverse_phi_table)):
        inverse_phi_table[i] = inverse_phi(Sigma[i])


    look_up_point = -1

    for i in range(len(inverse_phi_table)):

        if(inverse_phi_table[i]>argument_phi):
            look_up_point = i
            break


    if(look_up_point == -1):
        print('Error: Could not estimate Sigma, Please increase Phi inverse function range')


    result = Sigma[look_up_point]


    # plt.plot(Sigma, inverse_phi_table)
    #
    #
    # plt.plot(argument_phi, Sigma[look_up_point],'ro')
    # plt.show()

    return result


def Variance_update(Z, Mean):

    argument_phi = 0

    for i in range(len(Z)):

        argument_phi = argument_phi + pow(Riemannian_distance(Z[i], Mean),2)

    argument_phi = argument_phi/len(Z)

    return phi(Z,Mean, argument_phi)

def Variance_update2 (Z, weight, weights, variance, variances, barycentre, barycentres):

    argument_phi = 0

    nominator = 0

    denominator = 0

    for i in range(len(Z)):

        nominator = nominator + omega_mu(Z[i], weight, weights, variance, variances, barycentre, barycentres)*pow(Riemannian_distance(Z[i], barycentre),2)

    denominator = N_mu(Z, weight, weights, variance, variances, barycentre, barycentres)

    argument_phi = nominator/denominator

    return phi(argument_phi)

def omega_mu(z, weight, weights, variance, variances, barycentre, barycentres):

    nominator = weight * Gaussian_PDF(z, barycentre, variance)

    denominator = 0

    for i in range(len(weights)):
        denominator = denominator + weights[i]*Gaussian_PDF(z, barycentres[i], variances[i])

    #print('\t\tOmega_mu', nominator/denominator)
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

    result = (1/Z)*math.exp( (-pow(Riemannian_distance(X, Mean),2))/(2*pow(Sigma,2)))

    return result



def Gaussian_Mixture (X, Mean, Sigma, weights):

    k = len(weights)
    pdf = 0

    for i in range(k):
        pdf = pdf + weights[i]*Gaussian_PDF(X,Mean[i], Sigma[i])

    return pdf