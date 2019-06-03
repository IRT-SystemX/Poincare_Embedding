import numpy as np
import pickle

def Get_matrix(filename):


    with open('Input/'+filename, 'rb') as f:
        A, B, centroids, data_label, Variances, truth, oiu  = pickle.load(f)
    f.close()


    A = np.array(A)
    A = A.astype(int)

    n = len(A)
    #print('La matrice A est:\n', A)
    #print('Length of data', n)

    return B