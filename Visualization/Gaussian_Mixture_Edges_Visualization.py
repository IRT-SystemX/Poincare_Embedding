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
from matplotlib.axes._axes import _log as matplotlib_axes_logger



def Plot_Gaussian_Mixture_Edges(Data, Means, Variances, Weights, labels, color_array, edges, output_filename, dimension_index):


    #Reduce the axis logger to Error to avoid getting errors with colors not being provided as arrays



    matplotlib_axes_logger.setLevel('ERROR')

    #Number of sample points to plot the Gaussian mixture model

    N = 100
    X = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(X, Y)

    X0, Y0, radius = 0.0, 0.0, 1

    r = np.sqrt((X - X0) ** 2 + (Y - Y0) ** 2)

    inside_disk = r < radius

    Z = np.zeros((N,N))
    #print('Len(Z)', len(Z))
    counter = 0
    probability_error = False
    f = open(output_filename+"/Computed_Points_Gaussian_Mixture.txt", "w+")

    for i in range(N):
        for q in range(N):
            if(inside_disk[i][q]== True):
                Z[i][q] = Gaussian_Mixture(X[i][q]+ 1j*Y[i][q], Means, Variances, Weights )
            else:
                Z[i][q] = 0

            if(Z[i][q]>1):
                probability_error = True
            f.write(str(Z[i][q])+' ')
            if (q == N-1):
                f.write('\n')
            #print('Counter', counter)
            counter = counter +1

    f.close()

    if(probability_error==True):
        print('Probability greater than 1 error')
    else:
        print('All computed probabilities are less than 1')


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True, cmap=cm.viridis)

    z_height_circle = -0.8

    #Poincaré circle
    p = Circle((0, 0), 1, edgecolor='b', lw=1, facecolor='none')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=z_height_circle, zdir="z")

    for i in range(0, len(edges[:, 0])):
        for j in range(i, len(edges[0, :])):
            if i != j:
                if edges[i, j] == 1:
                    ax.plot([Data[i].real, Data[j].real], [Data[i].imag, Data[j].imag], z_height_circle, color='k', alpha=0.4)

    for i in range(len(Data)):
         ax.scatter(Data[i].real, Data[i].imag,z_height_circle, c = color_array[labels[i]], marker='.')

    for j in range(len(Means)):
        ax.scatter(Means[j].real, Means[j].imag, z_height_circle, c = color_array[j], marker = 'D')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.8, 0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')

    filename = output_filename+'Gaussian_mixture_edges_plot_dimension_'+str(dimension_index)+'.pdf'

    plt.savefig(filename, bbox_inches='tight')

    print('Gaussian with edges saved to',filename)

    #plt.show()