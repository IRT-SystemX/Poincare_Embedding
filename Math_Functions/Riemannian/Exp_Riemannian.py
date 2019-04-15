import numpy as np
import cmath



def Exp_Riemannian(z,v):
    v_square = abs(v)/(1-abs(z)*abs(z))

    theta = np.angle(v)

    numerator = (z + cmath.exp(1j * theta)) * cmath.exp(2 * v_square) + (z - cmath.exp(1j * theta))
    denominator = (1 + z.conjugate() * cmath.exp(1j * theta)) * cmath.exp(2 * v_square) + (1 - z.conjugate() * cmath.exp(1j * theta))
    result1 = numerator / denominator

    result = result1.real + result1.imag * 1j

    return result


def Exp_Riemannian_perturbation(z,v):
    perturbation = 1e-5
    v_square = abs(v)/(1-abs(z)*abs(z)+perturbation)

    theta = np.angle(v)

    numerator = (z + cmath.exp(1j * theta)) * cmath.exp(2 * v_square) + (z - cmath.exp(1j * theta))
    denominator = (1 + z.conjugate() * cmath.exp(1j * theta)) * cmath.exp(2 * v_square) + (1 - z.conjugate() * cmath.exp(1j * theta)) + perturbation
    result1 = numerator / denominator

    result = result1.real + result1.imag * 1j

    return result