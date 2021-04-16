from __future__ import division
import numpy as np
import os
from tqdm import tqdm
from main import main
import utilidades
import plot as pp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import lattice


def analisis_intensidad(samples, num_rayos, n0s, n1s, Lx, Ly, nx, ny, plotting, reload, folder):
    # Comprobamos que las dimensiones son las adecuadas
    assert n0s.shape == n1s.shape
    for n0, n1 in zip(n0s, n1s):
        if not reload:
            print(f'Realizando la simulación con {n0}, {n1} \n')
            for i in range(samples):
                successful = False
                while not successful:
                    print('sample -->', i, 'of', samples)
                    red = main(num_rayos, Lx, Ly, nx, ny, n0, n1, None, None, 1)
                    if not red.error:
                        if i == 0:
                            x, y = red.intensities.shape
                            acum = np.zeros((x, y, samples))
                        acum[:, :, i] = red.intensities
                        successful = True

            print('Guardando...')
            np.save(folder / f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}', acum)

            if plotting:
                print('Plotting')
                plot = pp.plot_class(red.coordx_rayos, red.coordy_rayos, red.idxx_rayos, red.idxy_rayos,
                                     red.dx, red.dy, red.nx, red.ny, red.n, folder=folder)
                # Gráfico de la intensidad media en la celda
                plot.plot_intensities(intensities=np.mean(acum, axis=2),
                                      name=f'mean_intensity_{n0}_{n1}', mean=True)
                # Gráfico de la std en la celda
                plot.plot_intensities(intensities=np.std(acum, axis=2),
                                      name=f'std_dev_{n0}_{n1}', std=True)


def analisis_longitudes(num_rayos, n0s, n1s, Lx, Ly, nx, ny, folder):
    '''
    Función para realizar el análisis de la relación entre la longitud del rayos
    y la distancia real recorrida.
    '''

    print('Analisis de la longitud del rayo vs longitud real recorrida')

    fig, ax = plt.subplots(8, 1, figsize=(8, 14))
    fig2, ax2 = plt.subplots(8, 1, figsize=(8, 14))

    for k, (n0, n1) in enumerate(zip(n0s, n1s)):

        # Solo hace falta realizar un sample. Estudiamos todos sus rayos
        red = main(num_rayos, Lx, Ly, nx, ny, n0, n1, None, None, 1)

        real_dist = []
        total_dist = []

        for i in range(num_rayos):
            dist = 0

            for j in range(len(red.coordx_rayos[i])-1):

                cx1glob, cy1glob = utilidades.extraer_coord_globales(
                    red.dx, red.dy, red.coordx_rayos[i][j], red.coordy_rayos[i][j], red.idxx_rayos[i][j], red.idxy_rayos[i][j])
                cx2glob, cy2glob = utilidades.extraer_coord_globales(
                    red.dx, red.dy, red.coordx_rayos[i][j+1], red.coordy_rayos[i][j+1], red.idxx_rayos[i][j+1], red.idxy_rayos[i][j+1])

                d = np.sqrt((cx2glob-cx1glob)**2 + (cy2glob-cy1glob)**2)
                dist += d

            cx0glob, cy0glob = utilidades.extraer_coord_globales(
                red.dx, red.dy, red.coordx_rayos[i][0], red.coordy_rayos[i][0], red.idxx_rayos[i][0], red.idxy_rayos[i][0])
            cxfglob, cyfglob = utilidades.extraer_coord_globales(
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
    fig.savefig(folder / 'dist_real_vs_dist_rayo.jpg')
    fig2.savefig(folder / 'hist_dist_real_vs_dist_rayo.jpg')


def analisis_punto_fijado(samples, n0, n1, nx, ny, folder):
    '''
    Estudio del histograma de intensidades para un punto fijado de la red.
    '''
    # Cargamos los valores de intensidad guardados para unos determinados índices
    acum = np.load(folder / f'intensities_{samples}_{n0}_{n1}.npy')

    # Mean and std
    i_mean = np.mean(acum, axis=2)
    i_std = np.std(acum, axis=2)

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
    plt.savefig(folder / 'punto_fijado.jpg')


def analisis_direccion_fija_diferente_ruido(samples, n0s, n1s, nx, ny, Lx, Ly, folder, chosen_theta):
    '''
    Estudio de intensidad y varianza vs distancia para diferentes niveles de ruido
    '''
    print(f'Dirección fija en {chosen_theta}')
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    fig2, ax2 = plt.subplots(2, 1, figsize=(8, 8))

    for n0, n1 in zip(n0s, n1s):
        acum = np.load(folder / f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')
        # Mean and std
        i_mean = np.mean(acum, axis=2)
        i_std = np.std(acum, axis=2)

        n0 = np.round(n0, 1)
        n1 = np.round(n1, 1)

        # Análsis para el theta_fijado
        d, i_mean_evol, i_std_evol, Iinverse = __calculate(
            i_mean=i_mean, i_std=i_std, Lx=Lx, Ly=Ly, n0=n0, n1=n1, nx=nx,
            ny=ny, chosen_theta=chosen_theta)

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
    fig.savefig(folder / 'intensidad_media_vs_d_diferente_ruido.jpg')
    fig2.savefig(folder / 'std_vs_d_diferente_ruido.jpg')


def analisis_ruido_fijo_diferente_direccion(samples, n0, n1, nx, ny, Lx, Ly, folder, chosen_theta):
    '''
    Estudio de intensidad y varianza vs distancia para diferentes direcciones y
    un mismo nivel de ruido
    '''
    print(f'Mismo ruido {np.round(n0,1)}, {np.round(n1,1)}, diferente dirección')
    acum = np.load(folder / f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')

    # Mean and std
    i_mean = np.mean(acum, axis=2)
    i_std = np.std(acum, axis=2)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    fig2, ax2 = plt.subplots(2, 1, figsize=(8, 8))

    for _ in range(20):

        d, i_mean_evol, i_std_evol, Iinverse = __calculate(
            i_mean, i_std, Lx, Ly, n0, n1, nx, ny, random_theta=True)

        ax[0].loglog(d, i_mean_evol, '-x', linewidth=1, label='<I>')
        ax2[0].loglog(d, i_std_evol**2, '-x', linewidth=1, label=r'$\sigma_I^2$')

        ax[1].semilogy(d, i_mean_evol, '-x', linewidth=1, label='<I>')
        ax2[1].semilogy(d, i_std_evol**2, '-x', linewidth=1, label=r'$\sigma_I^2$')

    for i in range(2):
        ax[i].set_xlabel('d')
        ax2[i].set_xlabel('d')
        ax[i].set_ylabel('<I>')
        ax2[i].set_ylabel(r'$\sigma_I^2$')

    fig.savefig(folder / 'i_vs_d_20_rayos.jpg')
    fig2.savefig(folder / 'sigma_vs_d_20_rayos.jpg')


def __calculate(i_mean, i_std, Lx, Ly, n0, n1, nx, ny, random_theta=False, chosen_theta=None):

    # para calcular una trayectoria podemos aprovechar las funciones ya descritas
    # con un índice de refracción homogéneo y un único rayo.

    red = main(num_rayos=1, Lx=Lx, Ly=Ly, nx=nx, ny=ny, n0=n0, n1=n1, loc_eje=None,
               A=None, caos=1, random_theta=random_theta, chosen_theta=chosen_theta)

    i_mean_evol = []
    i_std_evol = []
    d = []
    for k, (i, j) in enumerate(zip(red.idxx_rayos[0], red.idxy_rayos[0])):

        i_mean_evol.append(i_mean[i, j])
        i_std_evol.append(i_std[i, j])

        # calculamos las coordendas globales
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

    # Solo incluimos a partir del primero
    d = d[1:]

    # Esperado por Ley Gauss
    Iinverse = d**(-1)

    # Normalizamos
    Iinverse = Iinverse/Iinverse[0]

    i_mean_evol = i_mean_evol[1:]/max(i_mean_evol[1:])
    i_std_evol = i_std_evol[1:]/max(i_std_evol)

    return d, i_mean_evol, i_std_evol, Iinverse
