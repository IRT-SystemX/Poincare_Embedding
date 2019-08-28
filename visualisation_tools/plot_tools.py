import matplotlib.pyplot as plt
import matplotlib as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.patches import Circle, PathPatch
import numpy as np
import torch 
from function_tools import modules
import math
pi_2_3 = pow((2*math.pi),2/3)
a_for_erf = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)

def erf_approx(x):
    return torch.sign(x)*torch.sqrt(1-torch.exp(-x*x*(4/np.pi+a_for_erf*x*x)/(1+a_for_erf*x*x)))



def weighted_gmm_pdf(w, z, mu, sigma, distance):
    # print(z.size())
    # print(z.size(0), len(mu), z.size(1))
    z_u = z.unsqueeze(1).expand(z.size(0), len(mu), z.size(1))
    # print(z_u.size())
    # print(mu.size())
    mu_u = mu.unsqueeze(0).expand_as(z_u)

    distance_to_mean = distance(z_u, mu_u)
    sigma_u = sigma.unsqueeze(0).expand_as(distance_to_mean)
    distribution_normal = torch.exp(-((distance_to_mean)**2)/(2 * sigma_u**2))
    zeta_sigma = pi_2_3 * sigma *  torch.exp((sigma**2/2) * erf_approx(sigma/math.sqrt(2)))

    return w.unsqueeze(0).expand_as(distribution_normal) * distribution_normal/zeta_sigma.unsqueeze(0).expand_as(distribution_normal) 


def subplot_embedding_distribution(ax, W, pi, mu, sigma,  labels=None, N=100, colors=None ):
    # plotting prior
    X = np.linspace(-1, 1 ,N)
    Y = np.linspace(-1, 1 ,N)
    X, Y = np.meshgrid(X, Y)
    # plotting circle
    X0, Y0, radius = 0, 0, 1
    r = np.sqrt((X - X0)**2 + (Y * Y0)**2)
    disc = r < 1

    Z = np.zeros((N, N))
    # compute the mixture 
    for z_index in range(len(Z)):
        x =  torch.cat((torch.FloatTensor(X[z_index]).unsqueeze(-1), torch.FloatTensor(Y[z_index]).unsqueeze(-1)), -1)
        zz = weighted_gmm_pdf(pi, x, mu, sigma, modules.hyperbolicDistance)
        zz[zz != zz ]= 0
        Z[z_index] = zz.sum(-1).numpy()

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True, cmap=plt.get_cmap("viridis"))    
    z_circle = -0.8
    p = Circle((0, 0), 1, edgecolor='b', lw=1, facecolor='none')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z = z_circle, zdir="z")

    for q in range(len(W)):
        if(colors is not None):
            ax.scatter(W[q][0].item(), W[q][1].item(), z_circle, c=[colors[q]], marker='.')            
        else:
            ax.scatter(W[q][0].item(), W[q][1].item(), z_circle, c='b', marker='.')
        #print('Print labels', labels[q])

    for j in range(len(mu)):
        ax.scatter(mu[j][0].item(), mu[j][1].item(), z_circle, c='r', marker='D')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.8, 0.4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')

def plot_embedding_distribution_multi(W, pi, mu, sigma, labels=None, N=100, colors=None, save_path="figures/default.pdf"):
    fig = plt.figure("Embedding-Distribution")
    border_size = (math.sqrt(len(W)+0.0))
    if(border_size != round(border_size)):
        border_size += 1
    for i in range(len(W)):
        ax = fig.add_subplot(border_size, border_size, i+1, projection='3d')
        subplot_embedding_distribution(ax, W[i], pi[i], mu[i], sigma[i], labels=labels, N=N, colors=colors)
    plt.savefig(save_path, format="pdf")

    return fig

def plot_embedding_distribution(W, pi, mu, sigma,  labels=None, N=100, colors=None):

    # TODO : labels colors
    if(labels is None):
        # color depending from gaussian prob
        pass
    else:
        # color depending from labels given
        pass

    # plotting prior
    X = np.linspace(-1, 1 ,N)
    Y = np.linspace(-1, 1 ,N)
    X, Y = np.meshgrid(X, Y)
    # plotting circle
    X0, Y0, radius = 0, 0, 1
    r = np.sqrt((X - X0)**2 + (Y * Y0)**2)
    disc = r < 1

    Z = np.zeros((N, N))
    # compute the mixture 
    for z_index in range(len(Z)):
        #    print(torch.Tensor(X[z_index]))
        x =  torch.cat((torch.FloatTensor(X[z_index]).unsqueeze(-1), torch.FloatTensor(Y[z_index]).unsqueeze(-1)), -1)
        zz = weighted_gmm_pdf(pi, x, mu, sigma, modules.hyperbolicDistance)
        zz[zz != zz ]= 0
        #    print(zz.size())
        #    print(zz)
        #    print(weighted_gmm_pdf(pi, mu, mu, sigma, modules.hyperbolicDistance))
        Z[z_index] = zz.sum(-1).numpy() 
    # print(Z.max())
    fig = plt.figure("Embedding-Distribution")
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True, cmap=plt.get_cmap("viridis"))    
    z_circle = -0.8
    p = Circle((0, 0), 1, edgecolor='b', lw=1, facecolor='none')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z = z_circle, zdir="z")

    for q in range(len(W)):
        if(colors is not None):
            ax.scatter(W[q][0].item(), W[q][1].item(), z_circle, c=[colors[q]], marker='.')            
        else:
            ax.scatter(W[q][0].item(), W[q][1].item(), z_circle, c='b', marker='.')
        #print('Print labels', labels[q])

    for j in range(len(mu)):
        ax.scatter(mu[j][0].item(), mu[j][1].item(), z_circle, c='r', marker='D')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.8, 0.4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')

    plt.savefig("figures/embeddings_karate_node_distrib.pdf", format="pdf")

    return fig