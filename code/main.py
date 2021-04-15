import numpy as np
import time
import lattice as lat
import plot as pp


def main(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje, A, caos, name, folder, statistics):
    t0 = time.time()
    red = lat.lattice(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje, A, caos)
    red.evolucion()
    print('Finished in: ', time.time()-t0)
    if not statistics:
        print('Plotting...')
        pp.plot_trajectories(red.coordx_rayos, red.coordy_rayos, red.idxx_rayos, red.idxy_rayos, red.dx, red.dy, red.nx, red.ny, red.n, name=name,
                             folder=folder)
        if name is not None:
            pp.plot_intensities(red.intensities, red.nx, red.ny,
                                name=name+'_intensities', folder=folder)
        else:
            pp.plot_intensities(red.intensities, red.nx, red.ny, name=name, folder=folder)

    return red


if __name__ == '__main__':

    num_rayos = 1000
    Lx = 1
    Ly = 1
    nx = 50
    ny = 50
    n0 = 2
    n1 = 1
    loc_eje = None
    A = 0.8
    caos = None
    folder = 'results'
    statistics = False

    main(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje, A=A, caos=caos,
         name='normal', folder=folder, statistics=statistics)
