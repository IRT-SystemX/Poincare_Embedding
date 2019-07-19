import numpy as np
from scipy.sparse import lil_matrix

#Random Walk Generator given an adjacency matrix A and the length of a walk nstep
#(No check for singleton nodes is implemented, assumption: all nodes are connected).

def randomwalks(A,nstep):

    n = len(A[0])

    R = np.zeros((n,nstep+1), dtype = int)

    R[:, 0] = list(range(0, n))

    sum = np.sum(A,axis = 0)


    T = A/sum

    TS = np.cumsum(T,axis = 0)

    T[A>0] = TS[A>0]

    indices = list(range(0,n))

    for t in range(0, nstep):

        S = lil_matrix( (n,n))

        for k in range(0,len(R[:,t])):
                S[R[k,t],indices[k]] = 1

        H = np.array(np.random.rand(n) <= T*S,dtype = int)

        snext = np.sum(np.cumprod(1-H,axis = 0),axis = 0)

        R[:,t+1] = snext

    return R