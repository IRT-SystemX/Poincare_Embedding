from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d

fig = plt.figure()
ax = plt.axes(projection='3d')

p = Circle((5, 5), 3)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")


plt.show()