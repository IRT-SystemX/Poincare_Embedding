import numpy as np
from scipy.sparse import lil_matrix

def randomwalks(A,nstep):

    #print('===========================================')
    #print('Début Random Walks')
    n = len(A[0])

    #print('Taille du premier élément de A\n',n)

    R = np.zeros((n,nstep+1), dtype = int)

    R[:, 0] = list(range(0, n))

    #print('La matrice R est:\n',R)
    #print('La premiere colonne de R est:',R[:,0])

    sum = np.sum(A,axis = 0)

    #print('L\'ensemble des sommes de chaque colonne de A sont:\n',sum)


    #La somme des colonnes ne doit pas être egal à 1
    #Car sinon on a un noeud qui n'est pas relié à aucun des autres noeuds.
    T = A/sum
    #print('Division de A par la somme de chaque colonne est T:\n',T)

    TS = np.cumsum(T,axis = 0)

    #print('La somme cumulative de T est TS:\n',TS)

    T[A>0] = TS[A>0]

    #print('La valeur de T après après qu\'on reprend les valeurs de TS là ou A est >0 sont:\n',T)

    indices = list(range(0,n))

    #print('La matrice R est:\n', R)


    #for t in range(0, 1):
    for t in range(0, nstep):
        #print('Iteration number is:',t)
        #S est une matrice de taille n*n

        #Est-ce que c'est necessaire d'utiliser sparse ou pas ?
        S = lil_matrix( (n,n))

        #print('La matrice S est\n',S)

        #print('Colonne de R considérée', R[:,t])
        #print('Len est ',len(R[:,t]))

        for k in range(0,len(R[:,t])):
                S[R[k,t],indices[k]] = 1

        #print('La matrice S après qu\'on remplie\n',S)

        #H est un vecteur resultant de la comparaison <= d'un vecteur aleatoire (a valeur entre 0 et 1) avec T*S

        #print('Vecteur aléatoire de taille n\n',np.random.rand(n))

        #print('Le produit de T par S est:\n',T*S)
        H = np.array(np.random.rand(n) <= T*S,dtype = int)

        #print('Les valeurs de H sont:\n',H)

        #print('Les valeurs de (1-H):\n',1-H)

        #print('Produit(1-H)\n', np.cumprod(1-H,axis = 0))

        #print('1-produit\n', np.cumprod(1-H, axis = 0))

        #print('Somme du produit de 1-H\n',np.sum(np.cumprod(1-H,axis = 0),axis = 0))

        snext = np.sum(np.cumprod(1-H,axis = 0),axis = 0)

        #snext = 1 + np.sum(np.cumprod(1 - np.array(np.random.rand(n) <= T*S,dtype = int)))

        #print('La valeur du snext est\n',snext)

        R[:,t+1] = snext

        #print('Matrice R après remplissage de la colonne',t+1,'est\n',R)

    #print('Fin Random Walks')
    #print('===========================================')

    return R