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

def Gaussian_Plot():


    N = 4
    X = np.linspace(-1.2, 1.2, N)
    Y = np.linspace(-1.2, 1.2, N)
    X, Y = np.meshgrid(X, Y)


    print(X)
    print(Y)



    Z = np.array([[0.5,-1.2,-1.2,-1.2],
        [-0.4,-0.4,-0.4, -0.4],
        [0.4,0.4,0.4,0.4],
        [1.2,1.2,1.2,-0.5]])

    print(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)

    #Poincaré circle
    p = Circle((0, 0), 1, edgecolor='b', lw=1, facecolor='none')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")



    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(0, 1.1)

    plt.show()


Mean = [0,0]
Sigma = 2

Gaussian_Plot()