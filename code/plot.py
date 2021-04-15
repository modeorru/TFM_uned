import matplotlib.pyplot as plt
import pylab
import os
import seaborn as sns
import numpy as np

def plot_trajectories(coordx_rayos, coordy_rayos, idxx_rayos, idxy_rayos, dx, dy, nx, ny, n, name=None, folder=None, cruces=None):

    plt.rcParams.update({'font.size': 16})

    x = []; y = []
    for i in range(len(coordx_rayos)):
        x.append(np.array(coordx_rayos[i]) + np.array(idxx_rayos[i])*dx)
        y.append(np.array(coordy_rayos[i]) + np.array(idxy_rayos[i])*dy)


    plt.figure(figsize=(10,8))
    for i in range(len(x)):
        plt.plot(x[i], y[i], c='k', linestyle='-', linewidth=1)

    x = np.linspace(0,1,nx+1)
    y = np.linspace(0,1,ny+1)
    plt.pcolormesh(y, x, n, shading='auto')

    plt.xlim((0,1))
    plt.ylim((0,1))

    pylab.pcolor(x, y, n, cmap='hsv', vmin=0, vmax=2)
    plt.colorbar()

    if cruces is not None and len(cruces) > 0:
        for i in cruces:
            plt.scatter(i[0], i[1], s=20, c='r')


    if name is not None:
        plt.savefig(os.path.join(folder, '{}.jpg'.format(name)))
    else:
        plt.show()

def plot_intensities(intensities, nx, ny, name=None, folder=None, std=False, mean=False):

    plt.rcParams.update({'font.size': 16})
    intensities = np.array(intensities)

    intensities /= np.amax(intensities)

    x = np.linspace(0,1,nx+1)
    y = np.linspace(0,1,ny+1)
    plt.figure(figsize=(10,8))
    plt.pcolormesh(y, x, intensities, shading='auto')

    plt.xlim((0,1))
    plt.ylim((0,1))

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
        plt.savefig(os.path.join(folder, '{}.jpg'.format(name)))
    else:
        plt.show()



