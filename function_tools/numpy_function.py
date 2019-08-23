import numpy as np
import math

def aphi(sigma):
    return math.pow(sigma,2)+math.pow(sigma,4) + \
        ( ( ((math.pow(sigma,3))*math.sqrt(2)*math.exp(-(math.pow(sigma,2))/2)))/(math.sqrt(math.pi)*math.erf(sigma/(math.sqrt(2)))) )

sigma_pos = np.linspace(0.01,6, 400) 
sigma_inverse = np.array([aphi(sigma_pos[i].item()) for i in range(len(sigma_pos))])

class RiemannianFunction(object):
    PI_CST = pow((2*math.pi),2/3)
    SQRT_CST = math.sqrt(2)
    ERF_CST = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)
 
    @staticmethod
    def riemannian_distance(x, y):
        a = abs((x-y)/(1-y.conjugate()*x))
        num = 1 + a
        den = 1 - a
        return 0.5 * np.log(num/den)
        # error function 
    @staticmethod
    def erf(x):
        return np.sign(x)*np.sqrt(1-np.exp(-x*x*(4/np.pi+RiemannianFunction.ERF_CST*x*x)/(1+RiemannianFunction.ERF_CST*x**2)))
    
    @staticmethod
    def normalization_factor(sigma):
        return RiemannianFunction.PI_CST * sigma * np.exp((sigma**2)/2)*RiemannianFunction.erf(sigma/(RiemannianFunction.SQRT_CST))

    @staticmethod
    def phi(value):
        return sigma_pos[(sigma_inverse>value).nonzero()[0][0]]
