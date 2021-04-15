from __future__ import division
import numpy as np
import os
from main import main
from tqdm import tqdm
import utilidades
import study_intensity
import plot as pp
import matplotlib.pyplot as plt
import seaborn as sns


num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje, A, caos, folder, name, statistics, samples = utilidades.read()
print('Simulando', num_rayos, 'en', samples, 'samples')

reload = True
folder = 'results'

n0s = np.ones(10)*1
n1s = np.arange(1, 3, 0.2)

assert n0s.shape == n1s.shape

if not reload:
    for n0, n1 in zip(n0s, n1s):
        for i in range(samples):
            successful = False
            while not successful:
                print('-->', i)
                red = main(num_rayos, Lx, Ly, nx, ny, n0, n1,
                           loc_eje, A, caos, name, folder, statistics)
                if not red.error:
                    if i == 0:
                        x, y = red.intensities.shape
                        acum = np.zeros((x, y, samples))
                    acum[:, :, i] = red.intensities
                    successful = True

        print('Saving...')
        #np.save(os.path.join(folder, 'intensities_{}_{}_{}'.format(samples, n0, n1)), acum)

        print('Plotting')
        #pp.plot_intensities(np.mean(acum, axis=2), red.nx, red.ny, name='mean_intensity_{}_{}'.format(n0, n1), folder=folder, mean=True)
        #pp.plot_intensities(np.std(acum, axis=2), red.nx, red.ny, name='std_dev_{}_{}'.format(n0, n1), folder=folder, std=True)


#########################################
######## ANÁLISIS ESTADÍSTICOS ##########
#########################################

punto_fijado = False
direccion_fijada = False
diferente_ruido = False
analisis_longitud = False
analisis_cruce = True

##################


def extraer_coord_globales(dx, dy, cx, cy, ix, iy):

    cxglob = ix*dx + cx
    cyglob = iy*dy + cy
    return cxglob, cyglob


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'


def onSegment(p, q, r):
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):

        # Clockwise orientation
        return 1
    elif (val < 0):

        # Counterclockwise orientation
        return 2
    else:

        # Colinear orientation
        return 0

# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.


def doIntersect(p1, q1, p2, q2):

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    # If none of the cases
    return False


if analisis_cruce:

    print('Analisis de los cruces entre rayos')

    n1s = np.arange(1.2, 2, 0.2)
    n0s = np.ones(len(n1s))*1

    fig, ax = plt.subplots(len(n1s), 1)

    for k, (n0, n1) in enumerate(zip(n0s, n1s)):

        print('-->', n0, n1, '\n')
        cruces = []
        d_cruces = []

        # Solo hace falta realizar un sample. Estudiamos todos sus rayos
        num_rayos = 100
        red = main(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje,
                   A, caos, name, folder, statistics=True)

        for i in tqdm(range(num_rayos)):
            for j in range(1, len(red.coordx_rayos[i])-1):
                cx0glob, cy0glob = extraer_coord_globales(
                    red.dx, red.dy, red.coordx_rayos[i][0], red.coordy_rayos[i][0], red.idxx_rayos[i][0], red.idxy_rayos[i][0])
                cx1glob, cy1glob = extraer_coord_globales(
                    red.dx, red.dy, red.coordx_rayos[i][j], red.coordy_rayos[i][j], red.idxx_rayos[i][j], red.idxy_rayos[i][j])
                cx2glob, cy2glob = extraer_coord_globales(
                    red.dx, red.dy, red.coordx_rayos[i][j+1], red.coordy_rayos[i][j+1], red.idxx_rayos[i][j+1], red.idxy_rayos[i][j+1])

                for i2 in range(i+1, num_rayos):  # miramos el resto de rayos
                    for j2 in range(1, len(red.coordx_rayos[i2])-1):
                        if (red.idxx_rayos[i2][j2] == red.idxx_rayos[i][j]) and (red.idxy_rayos[i2][j2] == red.idxy_rayos[i][j]):
                            cx1glob2, cy1glob2 = extraer_coord_globales(
                                red.dx, red.dy, red.coordx_rayos[i2][j2], red.coordy_rayos[i2][j2], red.idxx_rayos[i2][j2], red.idxy_rayos[i2][j2])
                            cx2glob2, cy2glob2 = extraer_coord_globales(
                                red.dx, red.dy, red.coordx_rayos[i2][j2+1], red.coordy_rayos[i2][j2+1], red.idxx_rayos[i2][j2+1], red.idxy_rayos[i2][j2+1])
                            p1 = Point(cx1glob, cy1glob)
                            q1 = Point(cx2glob, cy2glob)
                            p2 = Point(cx1glob2, cy1glob2)
                            q2 = Point(cx2glob2, cy2glob2)

                            if doIntersect(p1, q1, p2, q2):
                                # we find the intersection point
                                found = False
                                iterat = 0
                                while iterat < 10:
                                    vec1 = np.array([cx2glob-cx1glob, cy2glob-cy1glob])
                                    [cx2globin, cy2globin] = [cx1glob, cy1glob] + 0.5*vec1
                                    p1 = Point(cx1glob, cy1glob)
                                    q1 = Point(cx2globin, cy2globin)
                                    if doIntersect(p1, q1, p2, q2):
                                        cx2glob = cx2globin
                                        cy2glob = cy2globin
                                    else:
                                        cx1glob = cx2globin
                                        cy1glob = cy2globin

                                    iterat += 1
                                cxcruce = cx1glob
                                cycruce = cy1glob

                                d = np.sqrt((cx0glob-cxcruce)**2 + (cy0glob-cycruce)**2)
                                d_cruces.append(d)
                                cruces.append([cxcruce, cycruce])

        # pp.plot_trajectories(red.coordx_rayos, red.coordy_rayos, red.idxx_rayos, red.idxy_rayos, red.dx, red.dy, red.nx, red.ny, red.n, name='estudio_cruces',
        #                      folder=folder, cruces=cruces)

        sns.histplot(d_cruces, binwidth=0.02,
                     ax=ax[k], label='{}-{}'.format(np.round(n0, 1), np.round(n1, 1)))
        ax[k].legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(folder, 'cruces_vs_dist.jpg'))

if analisis_longitud:

    print('Analisis de la longitud del rayo vs longitud real recorrida')

    n1s = np.arange(1.2, 2, 0.1)
    n0s = np.ones(len(n1s))*1
    num_rayos = 500
    fig, ax = plt.subplots(8, 1, figsize=(8, 14))
    fig2, ax2 = plt.subplots(8, 1, figsize=(8, 14))

    for k, (n0, n1) in enumerate(zip(n0s, n1s)):

        # Solo hace falta realizar un sample. Estudiamos todos sus rayos
        red = main(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje,
                   A, caos, name, folder, statistics=True)

        real_dist = []
        total_dist = []

        for i in range(num_rayos):
            dist = 0

            for j in range(len(red.coordx_rayos[i])-1):

                cx1glob, cy1glob = extraer_coord_globales(
                    red.dx, red.dy, red.coordx_rayos[i][j], red.coordy_rayos[i][j], red.idxx_rayos[i][j], red.idxy_rayos[i][j])
                cx2glob, cy2glob = extraer_coord_globales(
                    red.dx, red.dy, red.coordx_rayos[i][j+1], red.coordy_rayos[i][j+1], red.idxx_rayos[i][j+1], red.idxy_rayos[i][j+1])

                d = np.sqrt((cx2glob-cx1glob)**2 + (cy2glob-cy1glob)**2)
                dist += d

            cx0glob, cy0glob = extraer_coord_globales(
                red.dx, red.dy, red.coordx_rayos[i][0], red.coordy_rayos[i][0], red.idxx_rayos[i][0], red.idxy_rayos[i][0])
            cxfglob, cyfglob = extraer_coord_globales(
                red.dx, red.dy, red.coordx_rayos[i][-1], red.coordy_rayos[i][-1], red.idxx_rayos[i][-1], red.idxy_rayos[i][-1])

            dxr = cxfglob - cx0glob
            dyr = cyfglob - cy0glob
            dist_real = np.sqrt(dxr**2 + dyr**2)

            real_dist.append(dist_real)
            total_dist.append(dist)

        n0 = np.round(n0, 1)
        n1 = np.round(n1, 1)

        # Plotting distancia real vs distancia rayo

        ax[k].plot(real_dist, real_dist, 'r')
        ax[k].scatter(real_dist, total_dist, marker='x', s=5, label='{}-{}'.format(n0, n1))

        ax[k].set_xlabel('Distancia real')
        ax[k].set_ylabel('Distancia rayo')
        ax[k].legend(title='n0-n1')
        ax[k].set_ylim([0, 3])

        # Graficamos el histograma de diferencias entre distancias

        real_dist = np.array(real_dist)
        total_dist = np.array(total_dist)
        dif = abs(total_dist-real_dist)/real_dist

        sns.histplot(dif, ax=ax2[k], binwidth=0.03, label='{}-{}'.format(n0, n1))
        ax2[k].set_xlim((0, 2))
        ax2[k].set_ylim((0, 120))
        ax2[k].legend(title='n0-n1')
        ax2[k].set_xlabel('Error absoluto relativo')

    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(os.path.join(folder, 'dist_real_vs_dist_rayo.jpg'))
    fig2.savefig(os.path.join(folder, 'hist_dist_real_vs_dist_rayo.jpg'))

if punto_fijado:

    filename = 'intensities_{}_{}_{}.npy'.format(samples, n0, n1)
    acum = np.load(os.path.join(folder, filename))

    # Mean and std
    i_mean = np.mean(acum, axis=2)
    i_std = np.std(acum, axis=2)

    print('Seleccionamos una posición en la red y hacemos un histograma de intensidad')
    print('Rayos lanzados desde', int(nx/2), int(ny/2))

    iniciox = int(nx/2)
    inicioy = int(ny/2)
    opciones = [[1, 0], [0, 1], [-1, 0], [0, -1]]

    fig, axs = plt.subplots(9, 4, figsize=(16, 10))
    for i in range(9):
        for j in range(4):
            a = iniciox + opciones[j][0] + (i+1)*5
            b = inicioy + opciones[j][1] + (i+1)*5
            #print('Punto ', a, b, i_mean[a,b])
            point = acum[a, b, :]
            sns.histplot(point, ax=axs[i, j])
            axs[i, j].axvline(i_mean[a, b], c='r', linewidth=3,
                              label='media en ({},{})'.format(a, b))
            axs[i, j].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'punto_fijado.jpg'))

###################

if direccion_fijada:

    print('Estudio de intensidad y varianza vs distancia...')

    if diferente_ruido:
        print('Realizamos el análisis para un rayo y diferentes niveles de ruido')

        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        fig2, ax2 = plt.subplots(2, 1, figsize=(8, 8))

        for n0, n1 in zip(n0s[1:], n1s[1:]):
            filename = 'intensities_{}_{}_{}.npy'.format(samples, n0, n1)
            acum = np.load(os.path.join(folder, filename))

            n0 = np.round(n0, 1)
            n1 = np.round(n1, 1)

            # Mean and std
            i_mean = np.mean(acum, axis=2)
            i_std = np.std(acum, axis=2)

            # Análsis para el theta_fijado
            d, i_mean_evol, i_std_evol, Iinverse = study_intensity.calculate(
                i_mean, i_std, Lx, Ly, nx, ny, chosen_theta=np.pi/6)

            ax[0].loglog(d, i_mean_evol, '-x', linewidth=1, label='{}-{}'.format(n0, n1))
            ax2[0].loglog(d, i_std_evol**2, '-x', linewidth=1, label='{}-{}'.format(n0, n1))

            ax[1].semilogy(d, i_mean_evol, '-x', linewidth=1, label='{}-{}'.format(n0, n1))
            ax2[1].semilogy(d, i_std_evol**2, '-x', linewidth=1, label='{}-{}'.format(n0, n1))

        for i in range(2):
            ax[i].set_xlabel('d')
            ax2[i].set_xlabel('d')
            ax[i].set_ylabel('<I>')
            ax2[i].set_ylabel(r'$\sigma_I^2$')
            ax[i].legend(title='n0-n1')
            ax2[i].legend(title='n0-n1')

        plt.tight_layout()
        fig.savefig(os.path.join(folder, 'intensidad_media_vs_d_diferente_ruido.jpg'))
        fig2.savefig(os.path.join(folder, 'std_vs_d_diferente_ruido.jpg'))

    else:
        print('Realizamos el análisis para diferentes direcciones y un mismo nivel de ruido')

        filename = 'intensities_{}_{}_{}.npy'.format(samples, n0, n1)
        acum = np.load(os.path.join(folder, filename))

        # Mean and std
        i_mean = np.mean(acum, axis=2)
        i_std = np.std(acum, axis=2)

        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        fig2, ax2 = plt.subplots(2, 1, figsize=(8, 8))

        for _ in range(20):

            d, i_mean_evol, i_std_evol, Iinverse = study_intensity.calculate(
                i_mean, i_std, Lx, Ly, nx, nyi, random_theta=True)

            ax[0].loglog(d, i_mean_evol, '-x', linewidth=1, label='<I>')
            ax2[0].loglog(d, i_std_evol**2, '-x', linewidth=1, label=r'$\sigma_I^2$')

            ax[1].semilogy(d, i_mean_evol, '-x', linewidth=1, label='<I>')
            ax2[1].semilogy(d, i_std_evol**2, '-x', linewidth=1, label=r'$\sigma_I^2$')

        for i in range(2):
            ax[i].set_xlabel('d')
            ax2[i].set_xlabel('d')
            ax[i].set_ylabel('<I>')
            ax2[i].set_ylabel(r'$\sigma_I^2$')

        #fig.savefig(os.path.join(folder, 'i_vs_d_20_rayos.jpg'))
        #fig2.savefig(os.path.join(folder, 'sigma_vs_d_20_rayos.jpg'))
