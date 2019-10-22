import math
import torch
from torch import nn
from function_tools import poincare_function as pf

# z and wik must have same dimenssion except if wik is not given
def barycenter(z, wik=None, lr=5e-3, tau=5e-3, max_iter=math.inf, distance=pf.distance, normed=False):

    if(wik is None):
        wik = 1.
        barycenter = z.mean(0, keepdim=True)

    else:
        # print("1",wik.size())
        wik = wik.unsqueeze(-1).expand_as(z)
        # print("2",wik.size())*
        # barycenter = z.mean(0, keepdim=True)
        barycenter = (z*wik).sum(0, keepdim=True)/wik.sum(0)
        #print(barycenter)
        # print()
    if(len(z) == 1):
        return z
    iteration = 0
    cvg = math.inf

    while(cvg>tau and max_iter>iteration):

        iteration+=1
        if(type(wik) != float):
            grad_tangent = 2 * pf.log(barycenter.expand_as(z), z) * wik 
            if((barycenter == barycenter).float().mean() != 1):
                print("\n\n At least one barycenter is Nan : ")
                print(barycenter)
                print(wik.sum(0))
                print(wik.mean(0))
                print(wik.sum(1))
                print(wik.mean(1))
                print(iteration)
                exit()
        else:
            grad_tangent = 2 * pf.log(barycenter.expand_as(z), z)
        
        #print(type(wik))
        if(normed):
            # print(grad_tangent.size())
            if(type(wik) != float):
                # print(wik.sum(0, keepdim=True))
                grad_tangent /= wik.sum(0, keepdim=True).expand_as(wik)
            else:
                grad_tangent /= len(z)
        cc_barycenter = pf.exp(barycenter, lr * grad_tangent.sum(0, keepdim=True))
        cvg = distance(cc_barycenter, barycenter).max().item()
        # print(cvg)
        barycenter = cc_barycenter
    if(type(wik) != float):
            # to debug ponderate version
            #print(cvg, iteration, max_iter) 
        pass

    return barycenter

def test():
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import numpy as np
    from itertools import product, combinations
    from mpl_toolkits.mplot3d import Axes3D

    X = torch.randn(100, 2)*0.5 +(torch.rand(1, 2).expand(100, 2) -0.5) * 3
    xn  = X.norm(2,-1)

    X[xn>1] /= ((xn[xn>1]).unsqueeze(-1).expand((xn[xn>1]).shape[0], 2) +1e-3)

    mu = barycenter(X)

    ax = plt.subplot()
    p = Circle((0, 0), 1, edgecolor='b', lw=1, facecolor='none')
    ax.add_patch(p)
    plt.scatter(X[:,0].numpy(), X[:,1].numpy())
    plt.scatter(mu[0,0].item(),mu[0,1].item(), label="Poincare barycenter",
                marker="s", c="red", s=100.)
    plt.scatter(X.mean(0)[0].item(), X.mean(0)[1].item(), label="Euclidean barycenter",
                marker="s", c="green", s=100.)
    plt.legend()
    plt.show()
    print("3D")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    X = torch.randn(100, 3)*0.3 +(torch.rand(1, 3).expand(100, 3) -0.5) *3
    xn  = X.norm(2,-1)

    X[xn>1] /= ((xn[xn>1]).unsqueeze(-1).expand((xn[xn>1]).shape[0], 3) +1e-3)

    mu = barycenter(X)



    ax.scatter(X[:,0].numpy(), X[:,1].numpy(), X[:,2].numpy())
    ax.scatter(mu[0,0].item(),mu[0,1].item(), mu[0,2].item(),label="Poincare barycenter",
               marker="s", c="red", s=100.)
    ax.scatter(X.mean(0)[0].item(), X.mean(0)[1].item(), X.mean(0)[2].item(),label="Euclidean barycenter",
               marker="s", c="green", s=100.)
    ax.legend()
    plt.show()



if __name__ == "__main__":
    test()
