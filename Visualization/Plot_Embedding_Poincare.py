import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def Plot_Embedding_Poincare_Multidim(data,filename,dimension, show = True):


    # Put true if you have latex
    plt.rc('text', usetex=False)

    plt.rc('font', family='serif')

    #Axes and sub-axes configuration
    fig, ax = plt.subplots()
    ax.set_xlim(left=-1.2, right=1.2)
    ax.set_ylim(bottom=-1.2, top=1.2)
    ax.set_aspect('equal', 'box')

    #Title and labels of axis

    plt.ylabel(r'Second dimension $x_1$', fontsize=16)
    plt.xlabel(r'First dimension $x_0$', fontsize=16)
    plt.title(r"Hyperbolic Poincaré Embedding",
              fontsize=20, color='black')


    #Plotting Poincaré disk border

    cercle = Circle(xy=[0, 0], radius=1, edgecolor='b', lw=1, facecolor='none')
    ax.add_artist(cercle)

    #Plotting data
    for i in range(len(data)):
        plt.plot(float(data[i][0]), float(data[i][1]), 'k.')

    plt.savefig(filename + "Poincare_Embedding_Dim_"+str(dimension)+".pdf", bbox_inches='tight')

    print('\t Data plot for Dimension '+str(dimension)+' saved to ',filename + "Poincare_Embedding_Dim_"+str(dimension)+".pdf")

    #Open  the interactive console for visualization
    if(show):
        plt.show()