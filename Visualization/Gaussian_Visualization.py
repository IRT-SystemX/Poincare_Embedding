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

    N = 260
    X = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(X, Y)


    X0, Y0, radius = 0.0, 0.0, 1

    r = np.sqrt( (X-X0)**2+(Y-Y0)**2)

    inside_disk = r <radius

    #print('X',X)
    #print('len(X)',len(X))
    #print('Y',Y)

    Z = np.zeros((N,N))
    #print('Len(Z)', len(Z))
    counter = 0
    f = open("Output\Computed_Points_Gaussian.txt", "w+")
    for i in range(N):
        for q in range(N):
            if(inside_disk[i][q]== True):
                Z[i][q] = Gaussian_PDF(X[i][q]+ 1j*Y[i][q], Mean, Sigma )

            else:
                Z[i][q] = 0
            f.write(str(Z[i][q]) + ' ')
            if (q == N-1):
                f.write('\n')
            #print('Counter', counter)
            counter = counter +1


    #print('Probability table', Z)
    fig = plt.figure()
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

    #plt.show()


