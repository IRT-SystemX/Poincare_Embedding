import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib import cm
# register Axes3D class with matplotlib by importing Axes3D
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
from Math_Functions.Riemannian.Gaussian_PDF import *

def Plot_Gaussian(Mean, Sigma, output_filename):

    N = 60
    X = np.linspace(-0.5, 0.5, N)
    Y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(X, Y)

    #print('X',X)
    #print('len(X)',len(X))
    #print('Y',Y)

    Z = np.zeros((N,N))
    #print('Len(Z)', len(Z))
    counter = 0
    for i in range(N):
        for q in range(N):
            #print(Z[i][q])
            Z[i][q] = Gaussian_PDF(X[i][q]+ 1j*Y[i][q], Mean, Sigma )
            #print('Counter', counter)
            counter = counter +1


    #print('Probability table', Z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True,
                    cmap=cm.viridis)

    #Poincaré circle
    p = Circle((0, 0), 1, edgecolor='b', lw=1, facecolor='none')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")


    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(0, 1.1)

    filename = 'Output/'+output_filename

    plt.savefig(filename, bbox_inches='tight')

    print('Data plot saved to ',filename)

    plt.show()


