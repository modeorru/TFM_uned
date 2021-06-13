from __future__ import division
import numpy as np
import os
from tqdm import tqdm
from main import main
import utilidades
import plot as pp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import lattice


def analisis_intensidad(samples, num_rayos, n0s, n1s, Lx, Ly, nx, ny, plotting, reload, folder,
                        loc_eje=None, A=None, caos=1, thetacs=None, corrections=None):
    # Comprobamos que las dimensiones son las adecuadas
    assert n0s.shape == n1s.shape
    for k, (n0, n1) in enumerate(zip(n0s, n1s)):
        if not reload:
            print(f'Realizando la simulación con {n0}, {n1} \n')
            for i in tqdm(range(samples)):
                successful = False
                while not successful:
                    print(f'sample --> {i+1} of {samples}')
                    red = main(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje, A, caos)
                    if not red.error:
                        if i == 0:
                            x, y = red.intensities.shape
                            acum = np.zeros((x, y, samples))
                            t_minimizante = np.zeros((x, y, samples))
                        acum[:, :, i] = red.intensities
                        t_minimizante[:, :, i] = red.t_minimizante
                        successful = True

            print('Guardando...')
            np.save(folder / f'intensities_{samples}_{np.round(n0,2)}_{np.round(n1,2)}', acum)
            np.save(folder / f't_min_{samples}_{np.round(n0,2)}_{np.round(n1,2)}', t_minimizante)

            if plotting:
                print('Plotting')
                plot = pp.plot_class(red.coordx_rayos, red.coordy_rayos, red.idxx_rayos, red.idxy_rayos,
                                     red.dx, red.dy, red.nx, red.ny, red.n, folder=folder)
                # Trayectorias
                if (loc_eje is not None) and (thetacs is not None):  # para incluir el ángulo crítico
                    thetac = thetacs[k]
                    correction = corrections[k]
                else:
                    thetac = None
                    correction = None
                print('Trajectories')
                plot.plot_trajectories(maxn=2, cruces=None, name=f'trajectory_{n0}_{n1}',
                                       correction=correction, thetac=thetac, xlim=red.Lx, ylim=red.Ly)
        else:
            acum = np.load(folder / f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')
            t_minimizante = np.load(
                folder / f't_min_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')

        xdim, ydim, samples = t_minimizante.shape
        tmean = np.zeros((xdim, ydim))
        tstd = np.zeros((xdim, ydim))
        Imean = np.zeros((xdim, ydim))
        Istd = np.zeros((xdim, ydim))

        for i in range(xdim):
            for j in range(ydim):
                Inonegative = acum[i, j][acum[i, j] > 0]
                Imean[i, j] = np.mean(Inonegative)
                Istd[i, j] = np.std(Inonegative)
                tnonegative = t_minimizante[i, j][t_minimizante[i, j] > 0]
                tmean[i, j] = np.mean(tnonegative)
                tstd[i, j] = np.std(tnonegative)

        # Gráfico de la intensidad media en la celda
        plot = pp.plot_class(None, None, None, None, None,
                             None, nx, ny, None, folder=folder)
        print('Intensidades...')
        plot.plot_intensities(intensities=Imean,
                              name=f'mean_intensity_{n0}_{n1}', mean=True)
        # Gráfico de la std en las celdas
        plot.plot_intensities(intensities=Istd,
                              name=f'std_dev_{n0}_{n1}', std=True)
        # Gráfico de los tiempos minimizantes
        plot.plot_t_min(tmin=tmean, std=False,
                        mean=False, name=f'tminimizante_{n0}_{n1}', vmax=400)


def analisis_intensidad_modulaciones_suaves(num_rayos, Lx, Ly, nx, ny, plotting, folder, As):
    # Comprobamos que las dimensiones son las adecuadas

    for k, A in enumerate(As):
        print(f'Realizando la simulación con {A} \n')
        red = main(num_rayos, Lx, Ly, nx, ny, 1, 1, None, A, None)
        if not red.error:
            acum = red.intensities
            t_minimizante = red.t_minimizante

        print('Guardando...')
        np.save(folder / f'intensities_{np.round(A,1)}', acum)
        np.save(folder / f't_min_{np.round(A,1)}', t_minimizante)

        if plotting:
            print('Plotting')
            plot = pp.plot_class(red.coordx_rayos, red.coordy_rayos, red.idxx_rayos, red.idxy_rayos,
                                 red.dx, red.dy, red.nx, red.ny, red.n, folder=folder)

            plot.plot_trajectories(
                cruces=None, name=f'trajectory_{A}', correction=None, thetac=None, minn=0, maxn=2)
            # Gráfico de la intensidad media en la celda
            print('Intensidades...')
            plot.plot_intensities(intensities=acum,
                                  name=f'mean_intensity_{A}', mean=True, correction=None, thetac=None)
            # Gráfico de los tiempos minimizantes
            plot.plot_t_min(tmin=t_minimizante, std=False,
                            mean=False, name=f'tminimizante_{A}')


def analisis_correlacion_longitud(num_rayos, n0s, n1s, Lx, Ly, nx, ny, folder, reload):
    '''
    Función para realizar el análisis de la relación entre la longitud del rayos
    y la distancia real recorrida.
    '''
    plt.rcParams.update({'font.size': 16})
    idx_tesela_list = np.array([[i, i] for i in np.linspace(nx/2+1, 3*nx/4, 10)], dtype=int)

    if not reload:
        samples = 10
        real_dist = np.zeros((len(n1s), len(idx_tesela_list)))
        mean_total_dist = np.zeros((len(n1s),  len(idx_tesela_list)))
        std_total_dist = np.zeros((len(n1s),  len(idx_tesela_list)))
        for k, (n0, n1) in enumerate(zip(n0s, n1s)):
            print('Índices', n0, n1)
            for w, idx_tesela in enumerate(idx_tesela_list):
                print('--idx tesela', w, idx_tesela)
                ds = []
                reals = []
                for s in range(samples):
                    # Solo hace falta realizar un sample. Estudiamos todos sus rayos
                    red = main(num_rayos, Lx, Ly, nx, ny, n0, n1, None, None, 1)
                    for i in range(num_rayos):
                        dist = 0
                        ids = np.stack((red.idxx_rayos[i], red.idxy_rayos[i]), axis=-1)
                        for el, ins in enumerate(ids):
                            if (ins == idx_tesela).all():
                                for j in range(el):
                                    cx1glob, cy1glob = utilidades.extraer_coord_globales(
                                        red.dx, red.dy, red.coordx_rayos[i][j], red.coordy_rayos[i][j], red.idxx_rayos[i][j], red.idxy_rayos[i][j])
                                    cx2glob, cy2glob = utilidades.extraer_coord_globales(
                                        red.dx, red.dy, red.coordx_rayos[i][j+1], red.coordy_rayos[i][j+1], red.idxx_rayos[i][j+1], red.idxy_rayos[i][j+1])
                                    d = np.sqrt((cx2glob-cx1glob)**2 + (cy2glob-cy1glob)**2)
                                    dist += d

                                cx0glob, cy0glob = utilidades.extraer_coord_globales(
                                    red.dx, red.dy, red.coordx_rayos[i][0], red.coordy_rayos[i][0], red.idxx_rayos[i][0], red.idxy_rayos[i][0])
                                dxr = cx2glob - cx0glob
                                dyr = cy2glob - cy0glob
                                dist_real = np.sqrt(dxr**2 + dyr**2)

                        if dist != 0:
                            ds.append(dist)
                            reals.append(dist_real)
                print(ds)
                mean_total_dist[k, w] = np.mean(ds)
                std_total_dist[k, w] = np.std(ds)
                real_dist[k, w] = np.mean(reals)
        np.save(folder / 'mean_total_dist', mean_total_dist)
        np.save(folder / 'std_total_dist', std_total_dist)
        np.save(folder / 'real_dist', real_dist)
    else:
        mean_total_dist = np.load(folder / 'mean_total_dist.npy')
        std_total_dist = np.load(folder / 'std_total_dist.npy')
        real_dist = np.load(folder / 'real_dist.npy')

    df = pd.DataFrame()
    df['n1'] = np.array([[round(i, 1)]*len(idx_tesela_list) for i in n1s]).flatten()
    df['L'] = mean_total_dist.flatten()
    df['deucl'] = real_dist.flatten()
    df['std'] = std_total_dist.flatten()
    sns.lineplot(x='deucl', y='L', hue='n1', palette='viridis', linewidth=3, legend='full', data=df)
    # sns.lineplot(x='deucl', y='std', hue='n1', palette='viridis',
    #             linewidth=3,  legend=False, data=df)
    plt.plot(np.arange(0, 35, 0.1), np.arange(0, 35, 0.1), linestyle='dashed', c='r')
    plt.legend(ncol=2, title='n1', fontsize=10)
    plt.xlabel(r'$d_{eucl.}$')
    # plt.ylabel(r'$\sigma_L$')
    plt.ylabel(r'$<L_{rayo}>$')
    plt.tight_layout()
    plt.savefig(folder / 'longitud_y_distancia.jpg')
    # plt.savefig(folder / 'std_longitud_y_distancia.jpg')


def analisis_longitudes(num_rayos, n0s, n1s, Lx, Ly, nx, ny, folder, reload):
    '''
    Función para realizar el análisis de la relación entre la longitud del rayos
    y la distancia real recorrida.
    '''

    print('Analisis de la longitud del rayo vs longitud real recorrida')

    real_dist = np.zeros((len(n1s), num_rayos))
    total_dist = np.zeros((len(n1s), num_rayos))

    if not reload:
        for k, (n0, n1) in enumerate(zip(n0s, n1s)):

            print('Índices', n0, n1)
            # Solo hace falta realizar un sample. Estudiamos todos sus rayos
            red = main(num_rayos, Lx, Ly, nx, ny, n0, n1, None, None, 1)
            load = False
            if not load:
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

                    real_dist[k, i] = dist_real
                    total_dist[k, i] = dist

        np.save(folder / 'real_dist', real_dist)
        np.save(folder / 'total_dist', total_dist)
    else:
        real_dist = np.load(folder / 'real_dist.npy')
        total_dist = np.load(folder / 'total_dist.npy')

    plt.rcParams.update({'font.size': 14})
    for i, n1 in enumerate(n1s):
        df = pd.DataFrame()
        rs = []
        ts = []
        ns = []
        for j in range(num_rayos):
            if 160 > total_dist[i, j] >= 50:
                rs.append(real_dist[i, j])
                ts.append(total_dist[i, j])
                ns.append(n1)
        df['d_eucl'] = rs
        df['r'] = ts
        df['n1'] = ns
        # h = sns.jointplot(data=df, y='d_eucl', x='r', kind="hist")
        h = sns.jointplot(data=df, y='d_eucl', x='r', kind="hist", ylim=(49, 71), xlim=(50, 160))
        x0, x1 = h.ax_joint.get_xlim()
        y0, y1 = h.ax_joint.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        h.ax_joint.plot(lims, lims, '-r')
        plt.title(f'$n1={np.round(n1,1)}$')
        h.set_axis_labels(r'$L_{rayo}$', r'$d_{eucl.}$')
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.9, top=0.93, bottom=0.15)
        print(f'Guardando en {folder}', f'diagram_{i}.jpg')
        plt.savefig(folder / f'diagram_{i}.jpg')

    fig, ax = plt.subplots(len(n0s), 1, figsize=(10, 16))
    fig2, ax2 = plt.subplots(len(n0s), 1, figsize=(10, 16))

    print('Plotting!')
    for k, (n0, n1) in enumerate(zip(n0s, n1s)):
        print(k)
        n0 = np.round(n0, 1)
        n1 = np.round(n1, 1)

        # Plotting distancia real vs distancia rayo

        ax[k].plot(real_dist[k], real_dist[k], 'r')
        ax[k].scatter(total_dist[k], real_dist[k], marker='x', s=5, label=f'{n1}')
        ax[k].set_xlim(xmin=45, xmax=160)
        ax[k].set_ylim(ymin=49, ymax=71)
        ax[k].set_xlabel(r'$L_{rayo}$')
        ax[k].set_ylabel(r'$d_{eucl.}$')
        ax[k].legend(title='n1', loc='upper right')

        # Graficamos el histograma de distancias reales
        sns.histplot(total_dist[k], ax=ax2[k], label=f'{n1}')
        # ax2[k].set_xlim((0, 2))
        # ax2[k].set_ylim((0, 120))
        ax2[k].legend(title='n1')
        ax2[k].set_xlabel(r'$L_{rayo}$')

    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(folder / 'dist_real_vs_dist_rayo.jpg')
    fig2.savefig(folder / 'hist_dist_real_vs_dist_rayo.jpg')


def analisis_punto_fijado_para_todo_n(samples, n0s, n1s, nx, ny, folder):
    '''
    Estudio del histograma de intensidades para un punto fijado de la red.
    '''
    print('Histogramas de intensidad para diferentes distancias')
    # Cargamos los valores de intensidad guardados para unos determinados índices
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(5, 2, figsize=(14, 13))
    fig2, ax2 = plt.subplots(5, 2, figsize=(14, 13))
    kwargs = dict(histtype='stepfilled', alpha=0.4, density=True, bins=40, ec="k")

    for n0, n1 in zip(n0s, n1s):

        n0 = np.around(n0, 1)
        n1 = np.around(n1, 1)
        print(n1)
        try:
            acum = np.load(folder / f'intensities_{samples}_{n0}_{n1}.npy')
        except FileNotFoundError:
            acum = np.load('..' / folder / f'intensities_{samples}_{n0}_{n1}.npy')

        # Mean and std
        i_mean = np.mean(acum, axis=2)
        i_std = np.std(acum, axis=2)

        iniciox = int(nx/2)
        inicioy = int(ny/2)

        acum /= acum[iniciox, inicioy, :]
        i_mean /= i_mean[iniciox, inicioy]

        r = 0
        jump = 2
        i1 = 0
        i2 = 0

        for i, k in enumerate(range(iniciox+5, int(3*nx/4), jump)):
            point = acum[k, k, :]
            r += jump*np.sqrt(2)
            ax[i1, i2].hist(point, range=(0.01, 0.15), label=n1, **kwargs)
            ax[i1, i2].set_xlabel('I')
            ax[i1, i2].set_ylabel('Población')
            ax[i1, i2].legend(ncol=3)
            ax[i1, i2].set_title(f'r={np.around(r, 1)}')
            if i2 < 1:
                i2 += 1
            else:
                i2 = 0
                i1 += 1

        r = 0
        jump = 2
        i1 = 0
        i2 = 0

        for i, k in enumerate(range(iniciox+5, int(3*nx/4), jump)):
            point = acum[k, iniciox, :]
            r += jump
            ax2[i1, i2].hist(point, range=(0.01, 0.25), label=n1, **kwargs)
            ax2[i1, i2].set_xlabel('I')
            ax2[i1, i2].set_ylabel('Población')
            ax2[i1, i2].legend(ncol=3)
            ax2[i1, i2].set_title(f'r={np.around(r, 1)}')
            if i2 < 1:
                i2 += 1
            else:
                i2 = 0
                i1 += 1

    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(folder / f'histograma_intensidad_en_diagonal_all_n.jpg')
    fig2.savefig(folder / f'histograma_intensidad_en_eje_all_n.jpg')


def analisis_punto_fijado(samples, n0, n1, nx, ny, folder):
    '''
    Estudio del histograma de intensidades para un punto fijado de la red.
    '''
    print('Histogramas de intensidad para diferentes distancias')
    # Cargamos los valores de intensidad guardados para unos determinados índices
    try:
        acum = np.load(folder / f'intensities_{samples}_{n0}_{n1}.npy')
    except FileNotFoundError:
        acum = np.load('..' / folder / f'intensities_{samples}_{n0}_{n1}.npy')

    # Mean and std
    i_mean = np.mean(acum, axis=2)
    i_std = np.std(acum, axis=2)

    iniciox = int(nx/2)
    inicioy = int(ny/2)

    acum /= acum[iniciox, inicioy, :]
    i_mean /= i_mean[iniciox, inicioy]

    n1 = np.around(n1, 1)

    r = 0
    jump = 2
    i1 = 0
    i2 = 0

    fig, ax = plt.subplots(5, 2, figsize=(8, 7))
    for i, k in enumerate(range(iniciox+5, int(3*nx/4), jump)):
        point = acum[k, k, :]
        r += jump*np.sqrt(2)
        ax[i1, i2].hist(point, range=(0.01, 0.2), bins=40, label=f'r={np.around(r, 1)}')
        ax[i1, i2].set_xlabel('I')
        ax[i1, i2].set_ylabel('Población')
        ax[i1, i2].legend()
        if i2 < 1:
            i2 += 1
        else:
            i2 = 0
            i1 += 1

    plt.tight_layout()

    try:
        plt.savefig(folder / f'histograma_intensidad_en_diagonal_{n1}.jpg')
    except FileNotFoundError:
        plt.savefig('..' / folder / f'histograma_intensidad_en_diagonal_{n1}.jpg')
    plt.close()

    r = 0
    jump = 2
    i1 = 0
    i2 = 0

    fig, ax = plt.subplots(5, 2, figsize=(8, 7))
    for i, k in enumerate(range(iniciox+5, int(3*nx/4), jump)):
        point = acum[k, iniciox, :]
        r += jump
        ax[i1, i2].hist(point, range=(0.01, 0.25), bins=40, label=f'r={np.around(r, 1)}')
        ax[i1, i2].set_xlabel('I')
        ax[i1, i2].set_ylabel('Población')
        ax[i1, i2].legend()
        if i2 < 1:
            i2 += 1
        else:
            i2 = 0
            i1 += 1

    plt.tight_layout()

    try:
        plt.savefig(folder / f'histograma_intensidad_en_eje_{n1}.jpg')
    except FileNotFoundError:
        plt.savefig('..' / folder / f'histograma_intensidad_en_eje_{n1}.jpg')
    plt.close()


def analisis_direccion_fija_diferente_ruido(samples, n0s, n1s, nx, ny, Lx, Ly, folder, chosen_theta):
    '''
    Estudio de intensidad y varianza vs distancia para diferentes niveles de ruido
    '''
    print(f'Dirección fija en {np.round(chosen_theta*180/np.pi)}')
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(7, 6))
    fig2, ax2 = plt.subplots(figsize=(7, 6))

    df = pd.DataFrame()

    for i, (n0, n1) in enumerate(zip(n0s, n1s)):
        if n0 != n1:
            try:
                acum = np.load(
                    folder / f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')
            except FileNotFoundError:
                acum = np.load('..' / folder /
                               f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')
        else:
            # acum = np.load(folder / f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')
            acum = np.load(folder / f'intensities_{1}_{np.round(n0,1)}_{np.round(n1,1)}.npy')

        # Mean and std
        i_mean = np.mean(acum, axis=2)
        i_std = np.std(acum, axis=2)

        # n0 = np.round(n0, 1)
        # n1 = np.round(n1, 1)
        n0 = np.round(n0, 2)
        n1 = np.round(n1, 2)

        # Análsis para el theta_fijado
        d, i_mean_evol, i_std_evol = __calculate(i_mean=i_mean, i_std=i_std,
                                                 Lx=Lx, Ly=Ly, n0=1, n1=1, nx=nx,
                                                 ny=ny, chosen_theta=chosen_theta)
        d = np.sort(d)
        i_mean_evol = np.array(i_mean_evol)[np.argsort(d)]
        i_std_evol = np.array(i_std_evol)[np.argsort(d)]

        if i == 0:
            Ds = d
            Ns = np.array([n1]*len(d))
            Is = i_mean_evol
            Std = i_std_evol
        else:
            Ds = np.concatenate((Ds, d))
            Ns = np.concatenate((Ns, np.array([n1]*len(d))))
            Is = np.concatenate((Is, i_mean_evol))
            Std = np.concatenate((Std, i_std_evol))

    df['n1'] = np.array(Ns)
    df['r'] = np.array(Ds)
    df['<I>'] = np.array(Is)
    df['std'] = np.array(Std)

    palette = "hsv"
    sns.lineplot(x='r', y='<I>', hue='n1', data=df, linewidth=2,
                 ax=ax, palette=palette, legend='full')
    sns.lineplot(x='r', y='std', hue='n1', data=df, linewidth=2,
                 ax=ax2, palette=palette, legend='full')
    # if chosen_theta == 0:
    #    y = np.exp(-d*0.05 - 0.7)
    # else:
    #    y = np.exp(-d*0.05 - 1.1)
    # ax.semilogy(d, y, '--', linewidth=2, c='k', label='slope=-1')
    y = (d/d[0])**(-1)
    ax.loglog(d, y, '--', linewidth=2, c='k', label='slope=-1')
    ax.set_ylabel('<I>', rotation='horizontal', horizontalalignment='right')
    ax2.set_ylabel(r'$\sigma$', rotation='horizontal', horizontalalignment='right')
    # Setting scales
    ax.set(xscale="log", yscale="log")
    ax2.set(xscale="log", yscale="log")
    # ax.legend(title='n1', fontsize=12, ncol=3, loc='upper right')
    ax.legend(title='n1', fontsize=12, ncol=1, loc='lower left')
    ax2.legend(title='n1', fontsize=14, ncol=3)

    fig.tight_layout()
    fig2.tight_layout()
    try:
        fig.savefig(folder / f'I_ruido_1000_{np.round(chosen_theta*180/np.pi)}.jpg')
        fig2.savefig(folder / f'S_ruido_1000_{np.round(chosen_theta*180/np.pi)}.jpg')
    except FileNotFoundError:
        fig.savefig('..' / folder / f'I_ruido_1000_{np.round(chosen_theta*180/np.pi)}.jpg')
        fig2.savefig('..' / folder / f'S_ruido_1000_{np.round(chosen_theta*180/np.pi)}.jpg')


def analisis_ruido_fijo_diferente_direccion(samples, n0, n1, nx, ny, Lx, Ly, folder, chosen_theta):
    '''
    Estudio de intensidad y varianza vs distancia para diferentes direcciones y
    un mismo nivel de ruido
    '''
    print(f'Mismo ruido {np.round(n0,1)}, {np.round(n1,1)}, diferente dirección')

    if np.round(n0, 1) == np.round(n1, 1):
        acum = np.load(folder / f'intensities_{1}_{np.round(n0,1)}_{np.round(n1,1)}.npy')
    else:
        acum = np.load(folder / f'intensities_{samples}_{np.round(n0,1)}_{np.round(n1,1)}.npy')

    # Mean and std
    i_mean = np.mean(acum, axis=2)
    i_std = np.std(acum, axis=2)

    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(7, 6))
    fig2, ax2 = plt.subplots(figsize=(7, 6))

    df = pd.DataFrame()

    for i, theta in enumerate(chosen_theta):

        d, i_mean_evol, i_std_evol = __calculate(
            i_mean, i_std, Lx, Ly, 1, 1, nx, ny, chosen_theta=theta)

        d = np.sort(d)
        i_mean_evol = np.array(i_mean_evol)[np.argsort(d)]
        i_std_evol = np.array(i_std_evol)[np.argsort(d)]

        if i == 0:
            Ds = d
            Thetas = np.array([theta]*len(d))
            Is = i_mean_evol
            Std = i_std_evol
        else:
            Ds = np.concatenate((Ds, d))
            Thetas = np.concatenate((Thetas, np.array([np.round(theta*180/np.pi, 1)]*len(d))))
            Is = np.concatenate((Is, i_mean_evol))
            Std = np.concatenate((Std, i_std_evol))

    df['theta'] = np.array(Thetas)
    df['r'] = np.array(Ds)
    df['<I>'] = np.array(Is)
    df['std'] = np.array(Std)

    palette = "hsv"
    sns.lineplot(x='r', y='<I>', hue='theta', data=df, linewidth=2,
                 ax=ax, palette=palette, legend='full')
    sns.lineplot(x='r', y='std', hue='theta', data=df, linewidth=2,
                 ax=ax2, palette=palette, legend='full')
    y = (d/d[0])**(-1)
    y2 = (d/d[0])**(-2/3)*np.exp(3)
    # y3 = (d/d[0])**(-1/3)
    # ax.loglog(d, y3, linestyle='dotted', linewidth=2, c='k', label='slope=-1/3')
    # ax.loglog(d, y2, linestyle='dashdot', linewidth=2, c='k', label='slope=-2/3')
    # y = np.exp(-d*0.05 - 1.1)
    ax.loglog(d, y, '--', linewidth=2, c='k', label='slope=-1')
    ax2.loglog(d, y2, '--', linewidth=2, c='k', label='slope=-2/3')

    ax.set_title(f"$n_1={np.round(n1, 1)}$")
    ax2.set_title(f"$n_1={np.round(n1, 1)}$")
    ax.set_ylabel('<I>', rotation='horizontal', horizontalalignment='right')
    ax2.set_ylabel(r'$\sigma$', rotation='horizontal', horizontalalignment='right')
    # Setting scales
    ax.set(xscale="log", yscale="log")
    ax2.set(xscale="log", yscale="log")
    if n1 < 1.0:
        loc = 'upper right'
    else:
        loc = 'lower left'
    ax.legend(title=r'$\theta$', fontsize=12, ncol=2, loc=loc)
    ax2.legend(title=r'$\theta$', fontsize=14, ncol=2)

    fig.tight_layout()
    fig2.tight_layout()

    fig.savefig(folder / f'I_direcciones_{np.round(n0,1)}_{np.round(n1,1)}.jpg')
    fig2.savefig(folder / f'S_direcciones_{np.round(n0,1)}_{np.round(n1,1)}.jpg')


def analisis_tiempo_minimizante(samples, n0s, n1s, nx, ny, Lx, Ly, folder, chosen_theta):
    '''
    Estudio de los tiempos minimizantes y varianza vs distancia para diferentes direcciones y
    un mismo nivel de ruido
    '''
    print(f'Estudio de tiempo minimizante en la dirección {chosen_theta}')

    plt.rcParams.update({'font.size': 18})

    fig, ax = plt.subplots(figsize=(12, 10))
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    colormap = plt.cm.hsv  # nipy_spectral, Set1,Paired
    colors = [colormap(i) for i in np.linspace(0, 1, len(n1s))]

    for k, (n0, n1) in enumerate(zip(n0s, n1s)):
        print('Índices ->', n0, n1)
        try:
            acum = np.load(folder / f't_min_{samples}_{np.round(n0,2)}_{np.round(n1,2)}.npy')
        except FileNotFoundError:
            acum = np.load('..' / folder / f't_min_{samples}_{np.round(n0,2)}_{np.round(n1,2)}.npy')

        # Necesitamos coger únicamente las teselas que se han visitado
        xdim, ydim, samples = acum.shape

        tiempos_teselas = np.zeros((xdim, ydim))
        std_teselas = np.zeros((xdim, ydim))
        for i in range(xdim):
            for j in range(ydim):
                nonegative = acum[i, j][acum[i, j] > 0]
                if len(nonegative) == 0 and i == 50 and j == 50:
                    tiempos_teselas[i, j] = 0
                    std_teselas[i, j] = 0
                else:
                    tiempos_teselas[i, j] = np.mean(nonegative)
                    std_teselas[i, j] = np.std(nonegative)

        d, t_mean_evol, t_std_evol = __calculate(tiempos_teselas, std_teselas, Lx, Ly, 1, 1,
                                                 nx, ny, chosen_theta=chosen_theta, normalizar=False)
        d = np.sort(d)
        t_mean_evol = np.array(t_mean_evol)[np.argsort(d)]
        t_std_evol = np.array(t_std_evol)[np.argsort(d)]

        # Plot lineal
        # ax.plot(d, t_mean_evol, '-', color=colors[k], linewidth=3, label=np.round(n1, 1))
        # ax.plot(d, t_std_evol, '-', color=colors[k], linewidth=3, label=np.round(n1, 1))

        # Plot log-log
        ax.loglog(d, t_mean_evol, '-', color=colors[k], linewidth=3, label=np.round(n1, 1))
        ax2.loglog(d, t_std_evol, '-', color=colors[k], linewidth=3, label=np.round(n1, 1))

    y0 = np.exp(np.log(d) - 0.3)
    y1 = np.exp(np.log(d) - 4)
    y2 = np.exp(np.log(d**(2/3)) - 1.5)
    y3 = np.exp(np.log(d**(2/3)) + 1.5)
    ax.loglog(d, y0, '--', linewidth=2, c='k', label='slope=1')
    ax2.loglog(d, y1, '-', color='k', linestyle='dashed', label='slope=1')

    if chosen_theta == 0:
        ax2.loglog(d, y2, '-', color='k', linestyle='dashdot', label='slope=2/3')
    else:
        ax2.loglog(d, y3, '-', color='k', linestyle='dashdot', label='slope=2/3')

    ax.set_xlabel('r')
    ax2.set_xlabel('r')
    ax.set_xlim(0, 71)
    # ax.set_ylim(0, 450)
    ax.set_ylabel(r'<$t_{min}$>')
    ax2.set_ylabel(r'$\sigma_t$')

    ax.legend(title=r'n1', ncol=3, fontsize=16)
    ax2.legend(title=r'n1', ncol=3, fontsize=16)

    ax.set_title(r'$\theta$= ' + f'{round(chosen_theta*180/np.pi)}')
    ax2.set_title(r'$\theta$= ' + f'{round(chosen_theta*180/np.pi)}')
    fig.savefig(folder / f'log_mean_t_minimizantes_{round(chosen_theta*180/np.pi)}.jpg')
    fig2.savefig(folder / f'log_std_t_minimizantes_{round(chosen_theta*180/np.pi)}.jpg')

    plt.close()


def analisis_cruces(reload, samples, num_rayos, n0s, n1s, nx, ny, Lx, Ly, folder):
    '''
    Estudio de los cruces que ocurren.
    '''

    print('Analisis de los cruces entre rayos')
    df = pd.DataFrame()
    ns = []
    dis = []
    ins = []
    if not reload:
        for s in range(samples):
            print(s)
            for k, (n0, n1) in enumerate(zip(n0s, n1s)):
                print('-->', n0, n1, '\n')

                cruces = []
                d_cruces = []
                red = main(num_rayos, Lx, Ly, nx, ny, n0, n1, None, None, 1)

                i0x, i0y = red.idxx_rayos[0][0], red.idxy_rayos[0][0]
                for i in range(num_rayos):
                    print(i)
                    for j in range(1, len(red.coordx_rayos[i])-1):
                        cx0glob, cy0glob = utilidades.extraer_coord_globales(
                            red.dx, red.dy, red.coordx_rayos[i][0], red.coordy_rayos[i][0], red.idxx_rayos[i][0], red.idxy_rayos[i][0])
                        cx1glob, cy1glob = utilidades.extraer_coord_globales(
                            red.dx, red.dy, red.coordx_rayos[i][j], red.coordy_rayos[i][j], red.idxx_rayos[i][j], red.idxy_rayos[i][j])
                        cx2glob, cy2glob = utilidades.extraer_coord_globales(
                            red.dx, red.dy, red.coordx_rayos[i][j+1], red.coordy_rayos[i][j+1], red.idxx_rayos[i][j+1], red.idxy_rayos[i][j+1])
                        for i2 in range(i+1, num_rayos):  # miramos el resto de rayos
                            for j2 in range(1, len(red.coordx_rayos[i2])-1):
                                if (red.idxx_rayos[i2][j2] == red.idxx_rayos[i][j]) and (red.idxy_rayos[i2][j2] == red.idxy_rayos[i][j]):
                                    cx1glob2, cy1glob2 = utilidades.extraer_coord_globales(
                                        red.dx, red.dy, red.coordx_rayos[i2][j2], red.coordy_rayos[i2][j2], red.idxx_rayos[i2][j2], red.idxy_rayos[i2][j2])
                                    cx2glob2, cy2glob2 = utilidades.extraer_coord_globales(
                                        red.dx, red.dy, red.coordx_rayos[i2][j2+1], red.coordy_rayos[i2][j2+1], red.idxx_rayos[i2][j2+1], red.idxy_rayos[i2][j2+1])
                                    p1 = Point(cx1glob, cy1glob)
                                    q1 = Point(cx2glob, cy2glob)
                                    p2 = Point(cx1glob2, cy1glob2)
                                    q2 = Point(cx2glob2, cy2glob2)
                                    if utilidades.doIntersect(p1, q1, p2, q2):
                                        in1, in2 = red.idxx_rayos[i][j], red.idxy_rayos[i][j]
                                        intensidad = red.intensities[in1,
                                                                     in2]/red.intensities[i0x, i0y]
                                        # we find the intersection point
                                        found = False
                                        iterat = 0
                                        while iterat < 10:
                                            vec1 = np.array([cx2glob-cx1glob, cy2glob-cy1glob])
                                            [cx2globin, cy2globin] = [cx1glob, cy1glob] + 0.5*vec1
                                            p1 = Point(cx1glob, cy1glob)
                                            q1 = Point(cx2globin, cy2globin)
                                            if utilidades.doIntersect(p1, q1, p2, q2):
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

                                        ns.append(n1)
                                        dis.append(d)
                                        ins.append(intensidad)

            plot = pp.plot_class(red.coordx_rayos, red.coordy_rayos, red.idxx_rayos, red.idxy_rayos,
                                 red.dx, red.dy, red.nx, red.ny, red.n, folder=folder)
            # plot.plot_trajectories(maxn=2, cruces=cruces, name=f'estudio_cruces_{np.round(n1, 1)}')

        ins = np.array(ins)
        ns = np.array(ns)
        dis = np.array(dis)

        np.save(folder / 'dis', dis)
        np.save(folder / 'ns', ns)
        np.save(folder / 'ins', ins)

    else:
        dis = np.load(folder / 'dis.npy')
        ns = np.load(folder / 'ns.npy')
        ins = np.load(folder / 'ins.npy')

    df['n1'] = np.round(ns, 1)
    df['r'] = dis
    df['I'] = ins

    f, axs = plt.subplots()
    sns.histplot(data=df, x='I', element="poly", hue='n1', ax=axs)
    plt.ylabel('Cruces')
    plt.savefig(folder / 'histogramas.jpg')
    plt.close()

    sns.histplot(x='r', y='n1', data=df,   bins=(12, 9),  cmap="YlGnBu", cbar=True,
                 cbar_kws={'label': 'cruces'})
    plt.tight_layout()
    plt.savefig(folder / 'cruces_vs_dist.jpg')
    plt.close()


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def __calculate(i_mean, i_std, Lx, Ly, n0, n1, nx, ny, random_theta=False,
                chosen_theta=None, normalizar=True):

    # para calcular una trayectoria podemos aprovechar las funciones ya descritas
    # con un índice de refracción homogéneo y un único rayo.

    red = main(num_rayos=1, Lx=Lx, Ly=Ly, nx=nx, ny=ny, n0=n0, n1=n1, loc_eje=None,
               A=None, caos=1, random_theta=random_theta, chosen_theta=chosen_theta)

    mean_evol = []
    std_evol = []
    d = []
    last_i = red.idxx_rayos[0][0]
    last_j = red.idxy_rayos[0][0]
    xs = []
    ys = []

    for k, (i, j) in enumerate(zip(red.idxx_rayos[0], red.idxy_rayos[0])):

        mean_evol.append(i_mean[j, i])
        std_evol.append(i_std[j, i])

        # calculamos las coordendas globales
        xloc = red.coordx_rayos[0][k]
        yloc = red.coordy_rayos[0][k]
        xglob = i*red.dx + xloc
        yglob = j*red.dy + yloc

        if k == 0:
            x0 = xglob
            y0 = yglob
            last_x = x0
            last_y = y0
        else:
            if i > last_i:
                xglob += red.dx/2
            elif i < last_i:
                xglob -= red.dx/2
            else:
                xglob = last_x
            if j > last_j:
                yglob += red.dy/2
            elif j < last_j:
                yglob -= red.dy/2
            else:
                yglob = last_y

            xs.append(xglob)
            ys.append(yglob)
            r = np.sqrt((x0-xglob)**2 + (y0-yglob)**2)
            d.append(r)

            last_i = i
            last_j = j
            last_x = xglob
            last_y = yglob

    d = np.array(d)[:-1]

    # Solo guardamos a partir del 2o valor, para evitar problemas de d=0
    mean_evol = mean_evol[1:-1]
    std_evol = std_evol[1:-1]
    if normalizar:
        mean_evol = mean_evol/max(mean_evol)
        # if max(std_evol) > 0.001:
        #    i_std_evol = std_evol/max(std_evol)

    return d, mean_evol, std_evol
