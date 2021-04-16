import matplotlib.pyplot as plt
import pylab
import os
import seaborn as sns
import numpy as np


class plot_class():
    def __init__(self, coordx_rayos, coordy_rayos, idxx_rayos, idxy_rayos, dx, dy, nx, ny, n, folder):

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

    def plot_trajectories(self, cruces=None, name=None):

        plt.rcParams.update({'font.size': 16})

        x = []
        y = []
        for i in range(len(self.coordx_rayos)):
            x.append(np.array(self.coordx_rayos[i]) + np.array(self.idxx_rayos[i])*self.dx)
            y.append(np.array(self.coordy_rayos[i]) + np.array(self.idxy_rayos[i])*self.dy)

        plt.figure(figsize=(10, 8))
        for i in range(len(x)):
            plt.plot(x[i], y[i], c='k', linestyle='-', linewidth=1)

        x = np.linspace(0, 100, self.nx+1)
        y = np.linspace(0, 100, self.ny+1)
        plt.pcolormesh(y, x, self.n, shading='auto')

        plt.xlim((0, 100))
        plt.ylim((0, 100))

        pylab.pcolor(x, y, self.n, cmap='hsv', vmin=1, vmax=3)
        plt.colorbar()

        if cruces is not None and len(cruces) > 0:
            for i in cruces:
                plt.scatter(i[0], i[1], s=20, c='r')

        if name is not None:
            plt.savefig(self.folder / f'{name}.jpg')
        else:
            plt.show()

    def plot_intensities(self, intensities=None, std=False, mean=False, name=None):
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
        intensities = np.array(intensities)

        intensities /= np.amax(intensities)

        x = np.linspace(0, 100, self.nx+1)
        y = np.linspace(0, 100, self.ny+1)
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(y, x, intensities, shading='auto')

        plt.xlim((0, 100))
        plt.ylim((0, 100))

        if std:
            vmin = None
            vmax = None
        elif mean:
            vmin = 0
            vmax = 0.3
        else:
            vmin = 0
            vmax = 0.5

        pylab.pcolor(x, y, intensities, cmap='twilight', vmin=vmin, vmax=vmax)
        plt.colorbar()
        if name is not None:
            plt.savefig(self.folder / f'{name}.jpg')
        else:
            plt.show()

    def plot_t_min(self, tmin=None, std=False, mean=False, name=None):
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

        pylab.pcolor(x, y, tmin, cmap='twilight', vmin=0, vmax=500)
        plt.colorbar()
        if name is not None:
            plt.savefig(self.folder / f'{name}.jpg')
        else:
            plt.show()
