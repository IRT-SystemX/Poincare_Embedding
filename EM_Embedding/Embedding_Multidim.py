import numpy as np
import collections
import math
import cmath

from Math_Functions.Riemannian.Exp_Riemannian import *
from Math_Functions.Riemannian.Log_Riemannian import *
from Math_Functions.Riemannian.Distance_Gradient import *
from Math_Functions.Riemannian.Distance_Riemannian import *
from EM_Embedding.Random_Walk import *

def Embedding_Multidim_function(A,nstep, nepoch,context, p_gradient,negsample, number_disks, Random_Walks = None):

    n = len(A[0])
    eta = 0.05
    initialisation = 0.1
    #Longueur de l'exploraton des chemins par randomwalks
    #nstep = 10
    #p_gradient = 2

    #Fenêtre pour voir ocmbien de voisin apres et avant on prend pour placer les noeuds
    #context = 5

    #negsample = 10
    #nepoch = 10

    assert(nstep >= context*2)

    R = randomwalks(A, nstep)

    if Random_Walks is not None:
        Random_Walks = R

    npair = len(R[:,0]) * ((nstep + 1) * 2 * context - context * (context + 1))

    #print('npair:\n',npair)

    nodepairs = np.zeros((npair,2),dtype = int)

    #print('nodepairs',nodepairs)

    #Try with ind = 0
    #ind = 1
    ind = 0
    for i in range(0,len(R[:,0])):
        for j in range (0, len(R[0])):

            #Try instead of max(1,j-context) to be max(0,j-context)
            for k in range(max(0, j - context), min(len(R[0]), j + context)):
            #for k in range( max(1,j-context), min( len(R[0]),j+context )):
                if k != j:
                    nodepairs[ind,:] = [R[i,j], R[i,k]]
                    ind = ind+1


    #nodepairs is a large set of pairs between a node and it's context-close neighbor
    #==> we can't find in nodepairs a node with a neighboor that is at a distance exceeding the context

    assert(len(nodepairs[:,0]) == npair)

    #print('Node pairs for training\n', nodepairs)

    np.random.shuffle(nodepairs)


    #print('Node pairs after shuffling\n',nodepairs)


    #print('Matrice R\n',R)

    R_vector_column = []

    #print('taille colonne',len(R[:,0]) )
    #print('taille ligne',len(R[0,:]))

    for i in range(0, len(R[0, :])):
        for j in range(0, len(R[:, 0])):
            R_vector_column.append(R[j,i])

    #print('R_vector_column\n',R_vector_column)

    T = collections.Counter(R_vector_column)

    #print('Compteur de fréquence\n', T)

    Frequency_table = []
    for k,v in T.items():
        Frequency_table.append([k,v/len(R_vector_column)])

    Frequency_table = np.array(Frequency_table)

    #print('Elements de T et leur fréquence\n',Frequency_table)

    unigram = Frequency_table[:,1]

    #print('Colonne des frequences',unigram)

    Pneg = np.power(unigram,3/4)

    Pneg = Pneg /np.sum(Pneg)

    #print('Vecteur Pneg\n',Pneg)

    #Verifier que random est entre 0 et 1

    Embedding_table = []

    for i in range(number_disks):

        Embedding_table.append(initialisation*(2*np.random.random((n, 2))-1))

    #B = initialisation*(2*np.random.random((n, 2))-1)

    #print('Matrice B\n',B)

    counter = 0
    for epoch in range (0,nepoch):
        print('Iteration:\t',counter)
        obj = 0
        for p in range (0,npair):
            i = nodepairs[p,0]
            j = nodepairs[p,1]

            #neg est un tableau de taille negsample qui est rempli par des entiers
            #de 1 jusqu'à n avec une distribution de probabilité donnée par Pneg

            #Verifier que Pneg est bien des probabilités entre 0 et 1 et leur somme et egale a 1

            #Il faut faire attention si l'intention est de generer de 0 a n ou de 1 à n  (dans matlab c'est de 1 à n)
            #En python random.choice genere de 0 à n

            neg= np.random.choice(n,negsample, p = Pneg)

            #print('Le vecteur neg est',neg)

            for dim_index in range(number_disks):

                #print('\t\t Dimension (i.e. current Poincaré disk number) ', dim_index)

                zi = Embedding_table[dim_index][i,:]
                zj = Embedding_table[dim_index][j,:]

                zicomplex = complex(zi[0], zi[1])
                zjcomplex = complex (zj[0],zj[1])

                dij = Riemannian_distance(zicomplex, zjcomplex)

                #v = - distance_gradient(zi,zj)*logsigmoid_derivate(-dij)

                v =  distance_gradient(zi, zj,p_gradient) * (-sigmoid(dij))

                v1 = Exp_Riemannian(zicomplex,eta*v)

                #print('v1 est:',v1)

                Embedding_table[dim_index][i,:] = [v1.real,v1.imag]


                obj = obj - np.log(sigmoid(-dij))
                vj = -distance_gradient(zj, zi,p_gradient) * sigmoid(dij);

                for h in range(0,len(neg)):

                    zh = Embedding_table[dim_index][neg[h],:]
                    zhcomplex = complex(zh[0], zh[1])
                    djh = Riemannian_distance(zjcomplex, zhcomplex)
                    va = distance_gradient(zj,zh,p_gradient)*sigmoid(-djh)
                    vj = vj + va


                v1j = Exp_Riemannian(zjcomplex, eta*vj)


                Embedding_table[dim_index][j,:] = [v1j.real, v1j.imag]

                for h in range(0,len(neg)):
                    zh = Embedding_table[dim_index][neg[h],:]
                    zhcomplex = complex(zh[0],zh[1])
                    djh = Riemannian_distance(zhcomplex, zjcomplex)
                    va = distance_gradient(zh,zj,p_gradient)*sigmoid(-djh)
                    v1a = Exp_Riemannian(zhcomplex, eta*va)
                    Embedding_table[dim_index][neg[h],:] = [v1a.real,v1a.imag]
                    obj = obj-np.log(sigmoid(djh))

        counter = counter+1
        #print('La matrice B est\n',B)

    return Embedding_table,R


def logsigmoid_derivate(x):
    return 1/(1+cmath.exp(x))

def sigmoid (x):
    return 1/(1+cmath.exp(-x))


def Facebook_Function_text_application(A,nstep, nepoch,context, p_gradient,negsample, Random_Walks = None):

    print('Text embedding\n')
    n = len(A[0])
    eta = 0.005
    initialisation = 0.0001
    #Longueur de l'exploraton des chemins par randomwalks
    #nstep = 10
    #p_gradient = 2

    #Fenêtre pour voir ocmbien de voisin apres et avant on prend pour placer les noeuds
    #context = 5

    #negsample = 10
    #nepoch = 10

    assert(nstep >= context*2)

    R = randomwalks(A, nstep)

    if Random_Walks is not None:
        Random_Walks = R

    npair = len(R[:,0]) * ((nstep + 1) * 2 * context - context * (context + 1))

    print('npair:\n',npair)

    nodepairs = np.zeros((npair,2),dtype = int)

    #print('nodepairs',nodepairs)

    #Try with ind = 0
    #ind = 1
    ind = 0
    for i in range(0,len(R[:,0])):
        for j in range (0, len(R[0])):

            #Try instead of max(1,j-context) to be max(0,j-context)
            for k in range(max(0, j - context), min(len(R[0]), j + context)):
            #for k in range( max(1,j-context), min( len(R[0]),j+context )):
                if k != j:
                    nodepairs[ind,:] = [R[i,j], R[i,k]]
                    ind = ind+1


    #nodepairs is a large set of pairs between a node and it's context-close neighbor
    #==> we can't find in nodepairs a node with a neighboor that is at a distance exceeding the context

    assert(len(nodepairs[:,0]) == npair)

    #print('Node pairs for training\n', nodepairs)

    np.random.shuffle(nodepairs)


    #print('Node pairs after shuffling\n',nodepairs)


    #print('Matrice R\n',R)

    R_vector_column = []

    #print('taille colonne',len(R[:,0]) )
    #print('taille ligne',len(R[0,:]))

    for i in range(0, len(R[0, :])):
        for j in range(0, len(R[:, 0])):
            R_vector_column.append(R[j,i])

    #print('R_vector_column\n',R_vector_column)

    T = collections.Counter(R_vector_column)

    #print('Compteur de fréquence\n', T)

    Frequency_table = []
    for k,v in T.items():
        Frequency_table.append([k,v/len(R_vector_column)])

    Frequency_table = np.array(Frequency_table)

    #print('Elements de T et leur fréquence\n',Frequency_table)

    unigram = Frequency_table[:,1]

    #print('Colonne des frequences',unigram)

    Pneg = np.power(unigram,3/4)

    Pneg = Pneg /np.sum(Pneg)

    #print('Vecteur Pneg\n',Pneg)

    #Verifier que random est entre 0 et 1

    B = initialisation*(2*np.random.random((n, 2))-1)

    print('Initial B', B)

    #print('Matrice B\n',B)

    counter = 0
    for epoch in range (0,nepoch):
        print('Iteration:\t',counter)
        obj = 0
        for p in range (0,npair):
            i = nodepairs[p,0]
            j = nodepairs[p,1]

            #neg est un tableau de taille negsample qui est rempli par des entiers
            #de 1 jusqu'à n avec une distribution de probabilité donnée par Pneg

            #Verifier que Pneg est bien des probabilités entre 0 et 1 et leur somme et egale a 1

            #Il faut faire attention si l'intention est de generer de 0 a n ou de 1 à n  (dans matlab c'est de 1 à n)
            #En python random.choice genere de 0 à n

            neg= np.random.choice(n,negsample, p = Pneg)

            #print('Le vecteur neg est',neg)

            zi = B[i,:]
            zj = B[j,:]

            zicomplex = complex(zi[0], zi[1])
            zjcomplex = complex (zj[0],zj[1])

            dij = Riemannian_distance(zicomplex, zjcomplex)

            #v = - distance_gradient(zi,zj)*logsigmoid_derivate(-dij)

            v =  distance_gradient(zi, zj,p_gradient)

            v1 = Exp_Riemannian(zicomplex,-eta*v)

            #print('v1 est:',v1)

            B[i,:] = [v1.real,v1.imag]


            #obj = obj - np.log(sigmoid(-dij))
            vj = distance_gradient(zj, zi,p_gradient);

            Aj = 0

            for q in range(0,len(neg)):
                za = B[neg[q],:]
                zacomplex = complex(za[0],za[1])
                daj= Riemannian_distance(zjcomplex,zacomplex)
                Aj = Aj+cmath.exp(-math.pow(daj ,p_gradient))
            #print('Update de zj')
            for h in range(0,len(neg)):
                #print(h,' ')
                zh = B[neg[h],:]
                zhcomplex = complex(zh[0], zh[1])
                djh = Riemannian_distance(zjcomplex, zhcomplex)

                vh1 = -distance_gradient(zj,zh,p_gradient)*cmath.exp(-math.pow(djh ,p_gradient))
                vj = vj + vh1/Aj


            v1j = Exp_Riemannian(zjcomplex, -eta*vj)


            B[j,:] = [v1j.real, v1j.imag]

            #print('Zj PART')

            #print('len neg', len(neg))
            #print('Update of Zh ')
            for h in range(0,len(neg)):
                #print(h, ' ')
                zh = B[neg[h],:]
                zhcomplex = complex(zh[0],zh[1])
                djh = Riemannian_distance(zhcomplex, zjcomplex)
                #print(djh,'\n')
                #print('Distance',djh)

                #print('Distance at',h,' is ',djh)
                va = -distance_gradient(zh,zj,p_gradient)*cmath.exp(-math.pow(djh ,p_gradient))

                #Normalize after or during the loop?

                va = va/Aj
                v1a = Exp_Riemannian(zhcomplex, -eta*va)
                B[neg[h],:] = [v1a.real,v1a.imag]
                #obj = obj-np.log(sigmoid(djh))

        counter = counter+1
        #print('La matrice B est\n',B)

    return B,R