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
                            t_minimizante = np.zeros((x, y, samples))
                        acum[:, :, i] = red.intensities
                        t_minimizante[:, :, i] = red.t_minimizante
                        successful = True
            print('Guardando...')
            np.save(folder / f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}', acum)
            np.save(folder / f't_min_{samples}_{np.round(n0,1)}_{np.round(n1,1)}', t_minimizante)

            if plotting:
                print('Plotting')
                plot = pp.plot_class(red.coordx_rayos, red.coordy_rayos, red.idxx_rayos, red.idxy_rayos,
                                     red.dx, red.dy, red.nx, red.ny, red.n, folder=folder)
                # Trayectorias
                plot.plot_trajectories(cruces=None, name=f'trajectory_{n0}_{n1}')
                # Gráfico de la intensidad media en la celda
                print('Intensidades...')
                plot.plot_intensities(intensities=np.mean(acum, axis=2),
                                      name=f'mean_intensity_{n0}_{n1}', mean=True)
                # Gráfico de la std en las celdas
                plot.plot_intensities(intensities=np.std(acum, axis=2),
                                      name=f'std_dev_{n0}_{n1}', std=True)
                # Gráfico de los tiempos minimizantes
                plot.plot_t_min(tmin=np.mean(t_minimizante, axis=2), std=False,
                                mean=False, name=f'tminimizante_{n0}_{n1}')


def analisis_longitudes(num_rayos, n0s, n1s, Lx, Ly, nx, ny, folder):
    '''
    Función para realizar el análisis de la relación entre la longitud del rayos
    y la distancia real recorrida.
    '''

    print('Analisis de la longitud del rayo vs longitud real recorrida')

    fig, ax = plt.subplots(len(n0s), 1, figsize=(8, 14))
    fig2, ax2 = plt.subplots(len(n0s), 1, figsize=(8, 14))

    for k, (n0, n1) in enumerate(zip(n0s, n1s)):

        print('Índices', n0, n1)
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
        ax[k].scatter(total_dist, real_dist, marker='x', s=5, label=f'{n0}-{n1}')
        ax[k].set_xlim((45, 200))
        ax[k].set_xlabel('Longitud de rayo')
        ax[k].set_ylabel('Distancia real')
        ax[k].legend(title='n0-n1')

        # Graficamos el histograma de distancias reales

        sns.histplot(total_dist, ax=ax2[k], label=f'{n0}-{n1}', )
        #ax2[k].set_xlim((0, 2))
        #ax2[k].set_ylim((0, 120))
        ax2[k].legend(title='n0-n1')
        ax2[k].set_xlabel('Longitud de rayo')

    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(folder / 'dist_real_vs_dist_rayo.jpg')
    fig2.savefig(folder / 'hist_dist_real_vs_dist_rayo.jpg')


def analisis_punto_fijado(samples, n0, n1, nx, ny, folder):
    '''
    Estudio del histograma de intensidades para un punto fijado de la red.
    '''
    # Cargamos los valores de intensidad guardados para unos determinados índices
    try:
        acum = np.load(folder / f'intensities_{samples}_{n0}_{n1}.npy')
    except FileNotFoundError:
        acum = np.load('..' / folder / f'intensities_{samples}_{n0}_{n1}.npy')

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
                              label=f'media en ({a},{b})')
            axs[i, j].legend()
    plt.tight_layout()
    try:
        plt.savefig(folder / 'punto_fijado.jpg')
    except FileNotFoundError:
        plt.savefig('..' / folder / 'punto_fijado.jpg')


def analisis_direccion_fija_diferente_ruido(samples, n0s, n1s, nx, ny, Lx, Ly, folder, chosen_theta):
    '''
    Estudio de intensidad y varianza vs distancia para diferentes niveles de ruido
    '''
    print(f'Dirección fija en {np.round(chosen_theta*180/np.pi)}')
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 12))

    for n0, n1 in zip(n0s, n1s):
        try:
            acum = np.load(folder / f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')
        except FileNotFoundError:
            acum = np.load('..' / folder /
                           f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')

        # Mean and std
        i_mean = np.mean(acum, axis=2)
        i_std = np.std(acum, axis=2)

        n0 = np.round(n0, 1)
        n1 = np.round(n1, 1)

        # Análsis para el theta_fijado
        d, i_mean_evol, i_std_evol = __calculate(
            i_mean=i_mean, i_std=i_std, Lx=Lx, Ly=Ly, n0=1, n1=1, nx=nx,
            ny=ny, chosen_theta=chosen_theta)

        d = np.sort(d)
        i_mean_evol = np.array(i_mean_evol)[np.argsort(d)]
        i_std_evol = np.array(i_std_evol)[np.argsort(d)]

        ax[0].loglog(d, i_mean_evol, '-', linewidth=1, label=f'{n0}-{n1}')
        ax2[0].loglog(d, i_std_evol, '-', linewidth=1, label=f'{n0}-{n1}')

        ax[1].semilogy(d, i_mean_evol, '-', linewidth=1, label=f'{n0}-{n1}')
        ax2[1].semilogy(d, i_std_evol, '-', linewidth=1, label=f'{n0}-{n1}')

    ax[0].loglog(d, (d/d[0])**(-1/3), '--', linewidth=1, label='slope=-1/3')
    ax[0].loglog(d, (d/d[0])**(-2/3), '--', linewidth=1, label='slope=-2/3')
    ax[0].loglog(d, (d/d[0])**(-1), '--', linewidth=1, label='slope=-1')

    for i in range(2):
        ax[i].set_xlabel('d')
        ax2[i].set_xlabel('d')
        ax[i].set_ylabel('<I>')
        ax2[i].set_ylabel(r'$\sigma_I$')
        ax[i].legend(title='n0-n1')
        ax2[i].legend(title='n0-n1')

    plt.tight_layout()
    try:
        fig.savefig(folder / f'I_ruido_{np.round(chosen_theta*180/np.pi)}.jpg')
        fig2.savefig(folder / f'S_ruido_{np.round(chosen_theta*180/np.pi)}.jpg')
    except FileNotFoundError:
        fig.savefig('..' / folder / f'I_ruido_{np.round(chosen_theta*180/np.pi)}.jpg')
        fig2.savefig('..' / folder / f'S_ruido_{np.round(chosen_theta*180/np.pi)}.jpg')


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

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 12))

    for theta in chosen_theta:

        d, i_mean_evol, i_std_evol = __calculate(
            i_mean, i_std, Lx, Ly, 1, 1, nx, ny, chosen_theta=theta)

        d = np.sort(d)
        i_mean_evol = np.array(i_mean_evol)[np.argsort(d)]
        i_std_evol = np.array(i_std_evol)[np.argsort(d)]

        ax[0].loglog(d, i_mean_evol, '-', linewidth=2, label=f'{np.round(theta*180/np.pi)}')
        ax2[0].loglog(d, i_std_evol, '-', linewidth=2, label=f'{np.round(theta*180/np.pi)}')

        ax[1].semilogy(d, i_mean_evol, '-', linewidth=2, label=f'{np.round(theta*180/np.pi)}')
        ax2[1].semilogy(d, i_std_evol, '-', linewidth=2, label=f'{np.round(theta*180/np.pi)}')

    ax[0].loglog(d, (d/d[0])**(-2/3), '--', linewidth=1, label='slope=-2/3')
    ax[0].loglog(d, (d/d[0])**(-1/3), '--', linewidth=1, label='slope=-1/3')
    ax[0].loglog(d, (d/d[0])**(-1), '--', linewidth=1, label='slope=-1')
    #dis = np.arange(10, 70, 0.1)
    #ax[1].semilogy(dis, 0.4*np.exp(-dis/20), '--', color='r', label='slope=-20')

    for i in range(2):
        ax[i].set_xlabel('d')
        ax2[i].set_xlabel('d')
        ax[i].set_ylabel('<I>')
        ax2[i].set_ylabel(r'$\sigma_I$')
        ax[i].legend(title=f'{np.round(n0,1)}-{np.round(n1,1)}')
        ax2[i].legend(title=f'{np.round(n0,1)}-{np.round(n1,1)}')

    fig.savefig(folder / f'I_direcciones_{np.round(n0,1)}_{np.round(n1,1)}.jpg')
    fig2.savefig(folder / f'S_direcciones_{np.round(n0,1)}_{np.round(n1,1)}.jpg')


def analisis_tiempo_minimizante(samples, n0s, n1s, nx, ny, Lx, Ly, folder, chosen_theta):
    '''
    Estudio de los tiempos minimizantes y varianza vs distancia para diferentes direcciones y
    un mismo nivel de ruido
    '''
    print(f'Estudio de tiempo minimizante en la dirección {chosen_theta}')

    fig, ax = plt.subplots(2, 1, figsize=(10, 14))

    for n0, n1 in zip(n0s, n1s):
        print('Índices ->', n0, n1)
        try:
            acum = np.load(folder / f't_min_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')
        except FileNotFoundError:
            acum = np.load('..' / folder / f't_min_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')

        # Necesitamos coger únicamente las teselas que se han visitado
        xdim, ydim, samples = acum.shape
        teselas_visitadas = []

        for i in range(xdim):
            for j in range(ydim):
                values = []
                for s in range(samples):
                    if acum[i, j, s] > 0:
                        values.append(acum[i, j, s])
                teselas_visitadas.append(values)

        tmean = np.zeros((xdim, ydim))  # tiempo minimizante medio
        tstd = np.zeros((xdim, ydim))  # desviacion estándar en el tiempo minimizante
        idx = 0
        for i in range(xdim):
            for j in range(ydim):
                tmean[i, j] = np.mean(teselas_visitadas[idx])
                tstd[i, j] = np.std(teselas_visitadas[idx])
                idx += 1

        # En caso de ser medio homogéneo elegimos la dirección 0 para
        # asegurarnos que los rayos pasan por todas las distancias hasta 0.5
        if n0 == n1:
            chosen_theta = 0

        d, i_mean_evol, i_std_evol = __calculate(tmean, tstd, Lx, Ly, 1, 1,
                                                 nx, ny, chosen_theta=chosen_theta, normalizar=False)
        d = np.sort(d)
        i_mean_evol = np.array(i_mean_evol)[np.argsort(d)]
        i_std_evol = np.array(i_std_evol)[np.argsort(d)]

        # Excluimos el último porque el rayo se queda en la misma celda que es la penúltima
        d = d[:-1]
        i_mean_evol = i_mean_evol[:-1]
        i_std_evol = i_std_evol[:-1]

        ax[0].plot(d, i_mean_evol, '-', linewidth=2, label=f'{np.round(n0,1)}-{np.round(n1,1)}')
        ax[1].plot(d, i_std_evol, '-', linewidth=2, label=f'{np.round(n0,1)}-{np.round(n1,1)}')

    #ax[0].plot(d, d, '--', linewidth=1, color='k', label='slope=1')

    ax[0].set_xlabel('d')
    ax[1].set_xlabel('d')
    ax[0].set_ylabel(r'$t_{min}$')
    ax[1].set_ylabel(r'$\sigma_t$')
    ax[0].legend(title=f'theta: {np.round(chosen_theta*180/np.pi)}')
    ax[1].legend(title=f'theta: {np.round(chosen_theta*180/np.pi)}')

    fig.savefig(folder / 't_minimizantes.jpg')


def __calculate(i_mean, i_std, Lx, Ly, n0, n1, nx, ny, random_theta=False,
                chosen_theta=None, normalizar=True):

    # para calcular una trayectoria podemos aprovechar las funciones ya descritas
    # con un índice de refracción homogéneo y un único rayo.

    red = main(num_rayos=1, Lx=Lx, Ly=Ly, nx=nx, ny=ny, n0=n0, n1=n1, loc_eje=None,
               A=None, caos=1, random_theta=random_theta, chosen_theta=chosen_theta)

    i_mean_evol = []
    i_std_evol = []
    d = []
    for k, (i, j) in enumerate(zip(red.idxx_rayos[0], red.idxy_rayos[0])):

        i_mean_evol.append(i_mean[j, i])
        i_std_evol.append(i_std[j, i])

        # calculamos las coordenadas de las teselas que cruzan
        #xglob = i*red.dx
        #yglob = j*red.dy

        # calculamos las coordendas globales
        xloc = red.coordx_rayos[0][k]
        yloc = red.coordy_rayos[0][k]
        xglob = i*red.dx + xloc
        yglob = j*red.dy + yloc

        if k == 0:
            x0 = xglob
            y0 = yglob

        r = np.sqrt((x0-xglob)**2 + (y0-yglob)**2)
        d.append(r)

    d = np.array(d)
    # Solo guardamos a partir del 2o valor, para evitar problemas de d=0
    d = d[1:]
    i_mean_evol = i_mean_evol[1:]
    i_std_evol = i_std_evol[1:]

    if normalizar:
        i_mean_evol = i_mean_evol/max(i_mean_evol)
        i_std_evol = i_std_evol/max(i_std_evol)

    return d, i_mean_evol, i_std_evol
