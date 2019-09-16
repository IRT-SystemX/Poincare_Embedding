import math
from function_tools import poincare_function as pf

# z and wik must have same dimenssion except if wik is not given
def barycenter(z, wik=None, lr=1e-3, tau=1e-3, max_iter=math.inf, distance=pf.distance):
    if(wik is None):
        wik = 1
    else:
        wik = wik.unsqueeze(-1).expand_as(z)
    if(z.dim()>1):
        barycenter = z.mean(0, keepdim=True)
    else:
        barycenter = z.mean(0, keepdim=True)
    # print("barycenter sizes")
    # print(barycenter.size())
    # print(z.size())

    iteration = 0
    cvg = math.inf
    while(cvg>tau and max_iter>iteration):
        iteration+=1
        grad_tangent = pf.log(barycenter.expand_as(z), z) * wik
        cc_barycenter = pf.exp(barycenter, lr * grad_tangent.mean(0))
        cvg = distance(cc_barycenter, barycenter).max().item()
        barycenter = cc_barycenter
    return barycenter