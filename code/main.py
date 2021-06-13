import numpy as np
import time
import lattice as lat
import plot as pp


def main(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje, A, caos, random_theta=None, chosen_theta=None):
    '''
    Función principal para las simulaciones. En ella se inicializa la red y se deja evolucionar
    hasta que todos los rayos han alcanzado algún extremo de la caja.

    Argumentos importantes:

        -> num_rayos: int
                número de rayos "lanzados".
        -> Lx: float
                longitud en el eje x.
        -> Ly: float
                longitud en el eje y.
        -> nx: int
                número de celdas en el eje x.
        -> ny: int
                número de celdas en el eje y.
        -> n0: float
                límite inferior para el índice de refracción.
        -> n1: float
                límite superior para el índice de refracción.
        -> loc_eje: int o None
                if None: no se introduce un eje de cambio de índice.
                if int: se introduce un eje que separa dos índices.
        -> A: float |A|< 1 or None
                se introducen modulaciones suaves en el índice de refracción de
                acuerdo a la ec.(1)
                        n_{ij} =1 + A·sin(2·pi·i/L)·sin(2·pi·j/L)       (1)
        -> caos: 1 or None
                if 1: el índice de refracción obedecerá ec.(2)
                            n = U[n0,n1]                                (2)
    '''
    # Iniciamos tiempo
    t0 = time.time()
    # Iniciamos la red
    red = lat.lattice(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje,
                      A, caos, random_theta, chosen_theta)
    # Evolucionamos la red
    red.evolucion()

    # Calculamos el tiempo empleado
    #print('Finished in: ', time.time()-t0)
    return red


if __name__ == '__main__':

    num_rayos = 1000
    Lx = 1
    Ly = 1
    nx = 50
    ny = 50
    n0 = 2
    n1 = 1
    loc_eje = 10
    A = None
    caos = None
    folder = 'results/eje_vertical'
    statistics = False

    main(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje, A=A, caos=caos,
         random_theta=None, chosen_theta=None)
