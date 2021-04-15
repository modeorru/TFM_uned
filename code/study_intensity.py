import numpy as np
import lattice as lat
import plot as pp
import matplotlib.pyplot as plt

def calculate(i_mean, i_std, Lx, Ly, nx, ny, random_theta=False, chosen_theta=None):

    # para calcular una trayectoria podemos aprovechar las funciones ya descritas
    # con un índice de refracción homogéneo

    red  = lat.lattice(1, Lx, Ly, nx, ny, 1, 1, None, None, None, random_theta=random_theta, chosen_theta=chosen_theta)
    red.evolucion()
    #pp.plot_trajectories(red.coordx_rayos, red.coordy_rayos, red.idxx_rayos, red.idxy_rayos, red.dx, red.dy, red.nx, red.ny, red.n, name='1_rayo',
    #                          folder='results')

    i_mean_evol = []; i_std_evol = []; d = []
    for k, (i,j) in enumerate(zip(red.idxx_rayos[0], red.idxy_rayos[0])):

        i_mean_evol.append(i_mean[i,j])
        i_std_evol.append(i_std[i,j])

        #calculamos las coordendas globales
        xloc = red.coordx_rayos[0][k]
        yloc = red.coordy_rayos[0][k]
        xglob = i*red.dx + xloc
        yglob = j*red.dx + yloc

        if k == 0:
            x0 = xglob
            y0 = yglob

        r = np.sqrt((x0-xglob)**2 + (y0-yglob)**2)
        d.append(r)

    d = np.array(d)
    #if not all(np.sort(d) == d):
    #    print(np.sort(d), d)

    # Solo incluimos a partir del primero
    d = d[1:]

    # Esperado por Ley Gauss
    Iinverse = d**(-1)

    #Normalizamos
    Iinverse = Iinverse/Iinverse[0]

    i_mean_evol = i_mean_evol[1:]/max(i_mean_evol[1:])
    i_std_evol = i_std_evol[1:]/max(i_std_evol)

    return d, i_mean_evol, i_std_evol, Iinverse

