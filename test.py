import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib import cm
# register Axes3D class with matplotlib by importing Axes3D
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
from Math_Functions.Riemannian.Gaussian_PDF import *

N = 20
X = np.linspace(-1, 1, N)
Y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(X, Y)

print('X', X)
print('Y', Y)

X0, Y0, radius = 0.0, 0.0, 1

r = np.sqrt((X - X0) ** 2 + (Y - Y0) ** 2)

inside_disk = r < radius

print('Inside', inside_disk)

Z = np.zeros((len(X), len(X)))

Mean = 0.2+0.2j

Sigma = 0.7

for i in range(len(X)):
     for q in range(len(Y)):
         # print(Z[i][q])
        if(inside_disk[i][q]==True):
            Z[i][q] = Gaussian_PDF(X[i][q] + 1j * Y[i][q], Mean, Sigma)
        else:
            Z[i][q] = 0


#print('Z', Z[inside_disk])



fig = plt.figure()
#
ax = fig.gca(projection='3d')
#
ax.plot_surface(X,Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True, cmap=cm.viridis)
#
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(0, 1.1)

plt.show()