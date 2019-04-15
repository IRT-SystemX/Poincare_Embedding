import numpy as np

def Log_Riemannian(z,y):

    #print('Log de ', z, y)

    result = complex(0,0)

    #Important à revoir
    if (1 - z.conjugate() * y) == 0:

        print('============================')
        print('ERREUR')
        print('============================')

    else:
        qn = (y-z)/(1-z.conjugate()*y)

    normev = (1-abs(z)*abs(z))*np.arctanh(abs(qn))

    # print('Norme de v', normev)
    # print('Norme partie 1', (1-abs(z)*abs(z)))
    # print('Norme partie 2', np.arctanh(abs(qn)))

    theta = np.angle(qn)

   # if(abs(y-z)<(1e-7)):

       # result = complex(0,0)

    #else:

    result = complex(normev*(np.cos(theta)),normev*(np.sin(theta)))

    #print('resultat du log',result)
    return result