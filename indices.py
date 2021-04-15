import numpy as np


def asignar(nx, ny, n0, n1, loc_eje=None, A=None, caos=None, L=1):
    '''
    Dadas las dimensiones de la celda y los límites de los índices
    de refracción, se especifica la estrategia en la que los índices
    se asignan a cada celda. 
    '''

    n = np.ones((nx, ny))*n0

    if loc_eje is not None:

        # ÍNDICES SEPARADOS POR UN EJE VERTICAL
        assert 0 <= loc_eje <= nx-1
        nx2 = [i for i in range(nx) if i > loc_eje]
        ny2 = list(range(ny))
        for i in nx2:
            for j in ny2:
                n[j, i] = n1
    elif A is not None:

        # MODULACIONES SUAVES
        print('Modulaciones suaves')
        for i in range(nx):
            for j in range(ny):
                n[j, i] = 1 + A*np.sin(2*np.pi*j/ny)*np.sin(2*np.pi*i/nx)
    elif caos is not None:
        # CAOS
        print('Estocasticidad en los índices')
        for i in range(nx):
            for j in range(ny):
                rdn = np.random.rand()
                n[j, i] = rdn*n1 + (1-rdn)*n0
    else:
        print('Homogeneous n=1 refraction index')
    return n
