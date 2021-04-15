import numpy as np
from glob import glob


def calcular_contactos(dx, dy, cx, cy, idxx, idxy, theta):
    ''' Conocidas las coordendas (cx,cy) en el interior de la celda,
    podemos encontrar la distancia a las esquinas para determinar
    el próximo contacto.

    dx -- discretization in x axis
    dy -- discretization in y axis
    cx -- coordenada x
    cy -- coordenada y
    idxx -- índice x de la celda
    idxy -- índice y de la celda
    '''

    # Extremo superior-derecha
    xext1 = dx
    yext1 = dy
    angulo1 = np.arctan(abs(yext1-cy)/abs(xext1-cx))

    # Extremo superior-izquierda
    xext2 = 0.0
    yext2 = dy
    angulo2 = np.pi - np.arctan(abs(yext2-cy)/abs(xext2-cx))

    # Extremo inferior-izquierda
    xext3 = 0.0
    yext3 = 0.0
    angulo3 = np.pi + np.arctan(abs(yext3-cy)/abs(xext3-cx))

    # Extremo inferior-derecha
    xext4 = dx
    yext4 = 0.0
    angulo4 = 2*np.pi - np.arctan(abs(yext4-cy)/abs(xext4-cx))

    assert angulo1 < angulo2 < angulo3 < angulo4

    if 0 <= theta and theta < np.pi/2:

        if theta <= angulo1:  # hacia la derecha
            ix, iy = idxx+1, idxy
            mov_type = 'r'
            # Las modificaciones sobre las coordenadas
            dx = dx - cx
            dy = np.tan(theta)*dx

        else:  # hacia arriba
            ix, iy = idxx, idxy+1
            mov_type = 'u'

            # Las modificaciones sobre las coordenadas
            dy = dy - cy
            dx = dy/np.tan(theta)

    elif np.pi/2 <= theta and theta < np.pi:

        if angulo1 < theta and theta <= angulo2:  # hacia arriba
            ix, iy = idxx, idxy+1
            mov_type = 'u'

            dy = dy - cy
            dx = - dy*np.tan(theta - np.pi/2)
        else:  # hacia izquierda
            ix, iy = idxx-1, idxy
            mov_type = 'l'

            dx = - cx
            dy = dx*np.tan(theta - np.pi)

    elif np.pi <= theta and theta < 3*np.pi/2:

        if angulo2 < theta and theta <= angulo3:  # hacia izquierda
            ix, iy = idxx-1, idxy
            mov_type = 'l'

            dx = - cx
            dy = dx*np.tan(theta - np.pi)

        else:  # hacia abajo
            ix, iy = idxx, idxy-1
            mov_type = 'd'

            dy = - cy
            dx = dy*np.tan(3*np.pi/2 - theta)

    else:

        if angulo3 < theta and theta <= angulo4:  # hacia abajo
            ix, iy = idxx, idxy-1
            mov_type = 'd'

            dy = - cy
            dx = -dy*np.tan(theta + 3*np.pi/2)

        else:  # hacia derecha
            ix, iy = idxx+1, idxy
            mov_type = 'r'
            dx = dx - cx
            dy = - dx/np.tan(theta - 3*np.pi/2)

    return dx, dy, ix, iy, mov_type


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


def read():
    filename = glob('inputs.txt')[0]

    with open(filename, 'r') as f:
        data = f.readlines()
        d = [i.split()[1] for i in data[2:]]

    d[7] = None if d[7] == 'None' else d[7]
    d[8] = None if d[8] == 'None' else d[8]
    d[9] = None if d[9] == 'None' else d[9]
    d[11] = None if d[11] == 'None' else d[11]
    d[12] = True if d[12] == 'True' else False

    info = [int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]), float(d[5]), float(d[6]),
            d[7], d[8], d[9], d[10], d[11], d[12], int(d[13])]
    return info
