import matplotlib.pyplot as plt
import pylab
import os
import seaborn as sns
import numpy as np


class plot_class():
    def __init__(self, coordx_rayos, coordy_rayos, idxx_rayos, idxy_rayos, dx,
                 dy, nx, ny, n, folder):

        self.coordx_rayos = coordx_rayos
        self.coordy_rayos = coordy_rayos
        self.idxx_rayos = idxx_rayos
        self.idxy_rayos = idxy_rayos
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.n = n
        self.folder = folder

    def plot_trajectories(self, cruces=None, name=None, correction=None, thetac=None, minn=1, maxn=3, xlim=100, ylim=100):

        plt.rcParams.update({'font.size': 25})
        x = []
        y = []
        for i in range(len(self.coordx_rayos)):
            x.append(np.array(self.coordx_rayos[i]) + np.array(self.idxx_rayos[i])*self.dx)
            y.append(np.array(self.coordy_rayos[i]) + np.array(self.idxy_rayos[i])*self.dy)

        plt.figure(figsize=(10, 8))

        for i in range(len(x)):
            plt.plot(x[i], y[i], c='k', linestyle='-', linewidth=3)

        # Plot del ángulo crítico (descomentar las líneas)
        #xmax = 70
        #ymax = xmax*np.tan(thetac)
        #xr = np.arange(0, xmax, 0.1)
        #yr = ymax - (ymax/xmax)*xr
        #plt.plot(xr, yr+correction, 'white', linewidth=4, linestyle='dashed')

        x = np.linspace(0, xlim, self.nx+1)
        y = np.linspace(0, ylim, self.ny+1)
        plt.pcolormesh(y, x, self.n, shading='auto')

        plt.xlim((0, xlim))
        plt.ylim((0, ylim))

        pylab.pcolor(x, y, self.n, cmap='hsv', vmin=minn, vmax=maxn)
        bar = plt.colorbar()
        bar.set_label('n', rotation=0, labelpad=20, fontsize=25)

        if cruces is not None and len(cruces) > 0:
            for i in cruces:
                plt.scatter(i[0], i[1], s=100, c='blue')

        if name is not None:
            plt.savefig(self.folder / f'{name}.jpg')
        else:
            plt.show()
        plt.close()

    def plot_intensities(self, intensities=None, std=False, mean=False, name=None, correction=None, thetac=None):
        '''
        Hacer un gráfico de las intensidades sobre las teselas de la red.

        Args:

            -> intensities: array [nx, ny]
                    valores de intensidad.
            -> std: bool
                    si hacemos el gráfico de la fluctuación de intensidad.
            -> mean: bool
                    si hacemos el gráfico de la intensidad media.
            -> name: str
                    nombre con el que se guarda el archivo.
        '''

        plt.rcParams.update({'font.size': 25})
        intensities = np.array(intensities)
        intensities = np.transpose(intensities)
        intensities /= np.amax(intensities)

        x = np.linspace(0, 100, self.nx+1)
        y = np.linspace(0, 100, self.ny+1)
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(y, x, intensities, shading='auto')

        # Plot del ángulo crítico (descomentar las líneas)
        if thetac is not None:
            xmax = 70
            ymax = xmax*np.tan(thetac)
            xr = np.arange(0, xmax, 0.1)
            yr = ymax - (ymax/xmax)*xr
            plt.plot(xr, yr+correction, 'white', linewidth=2, linestyle='dashed')

        plt.xlim((0, 100))
        plt.ylim((0, 100))

        if std:
            vmin = None
            vmax = None
        elif mean:
            vmin = 0
            vmax = 0.1
        else:
            vmin = 0
            vmax = 0.5

        pylab.pcolor(y, x, intensities, cmap='twilight', vmin=vmin, vmax=vmax)
        bar = plt.colorbar()
        bar.set_label('I', rotation=0, labelpad=20, fontsize=25)

        if name is not None:
            plt.savefig(self.folder / f'{name}.jpg')
        else:
            plt.show()
        plt.close()

    def plot_t_min(self, tmin=None, std=False, mean=False, name=None, vmax=200):
        '''
        Hacer un gráfico de las intensidades sobre las teselas de la red.

        Args:

            -> intensities: array [nx, ny]
                    valores de intensidad.
            -> std: bool
                    si hacemos el gráfico de la fluctuación de intensidad.
            -> mean: bool
                    si hacemos el gráfico de la intensidad media.
            -> name: str
                    nombre con el que se guarda el archivo.
        '''

        plt.rcParams.update({'font.size': 16})
        tmin = np.array(tmin)

        x = np.linspace(0, 100, self.nx+1)
        y = np.linspace(0, 100, self.ny+1)
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(y, x, tmin, shading='auto')

        plt.xlim((0, 100))
        plt.ylim((0, 100))

        #pylab.pcolor(x, y, tmin, cmap='twilight', vmin=0, vmax=500)
        pylab.pcolor(y, x, tmin, cmap='twilight', vmin=0, vmax=vmax)
        bar = plt.colorbar()
        bar.set_label(r'$t_{min}$', rotation=0, labelpad=20, fontsize=16)

        if name is not None:
            plt.savefig(self.folder / f'{name}.jpg')
        else:
            plt.show()

        plt.close()
