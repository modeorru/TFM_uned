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
import estudios_estadisticos


class estadisticas():
    def __init__(self, samples, num_rayos, Lx, Ly, nx, ny, loc_eje, folder):
        '''
        Realizamos estudios estadisticos lanzando "num_rayos" tantas veces como "samples".
        Argumentos importantes:

            -> samples: int
                    número de simulaciones
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
            -> folder: str
                    donde guardar los resultados
        '''
        self.samples = samples
        self.num_rayos = num_rayos
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.loc_eje = loc_eje
        self.folder = folder

    #########################################
    ######## ANÁLISIS ESTADÍSTICOS ##########
    #########################################

    def Analisis_Intensidad(self, n0s, n1s, plotting, reload, thetacs, corrections):
        estudios_estadisticos.analisis_intensidad(self.samples, self.num_rayos, n0s, n1s, self.Lx, self.Ly,
                                                  self.nx, self.ny, plotting, reload, self.folder, self.loc_eje,
                                                  None, None, thetacs, corrections)

    def Plot_I_decay(self):
        n0 = 1.0
        n1 = 1.0
        intensity = np.load(
            folder / f'intensities_1_{np.round(n0, 1)}_{np.round(n1, 1)}.npy').squeeze()

        idx = 50
        intensity = intensity[idx+1:, idx]/max(intensity[idx+1:, idx])
        x = np.linspace(1, Lx/2, 49)
        I = x**-1
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({'font.size': 20})
        plt.loglog(x, intensity, label='numérico', c='red', linewidth=3)
        plt.loglog(x, I/I[0], label=r'$r^{-1}$',
                   linewidth=2, color='k', linestyle='dashed')
        plt.ylabel('I', rotation='horizontal', horizontalalignment='right')
        plt.xlabel('r')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.folder / 'power_law_isotropico.jpg')


if __name__ == '__main__':

    # Definimos las variables a usar
    samples = 1
    num_rayos = 10
    Lx = 100
    Ly = 100
    nx = 100
    ny = 100
    loc_eje = 69
    folder = Path('results/eje_vertical/metodos')
    folder.mkdir(exist_ok=True)

    # CREAMOS LA CLASE ESTADISTICAS
    stats = estadisticas(samples, num_rayos, Lx, Ly, nx, ny, loc_eje, folder)

    # ANÁLISIS INTENSIDAD Y STD EN LA RED

    n1s = np.arange(1, 3, 0.2)
    n1s = np.arange(1, 2, 0.2)
    n0s = np.ones(len(n1s))*1
    thetacs = None
    corrections = None

    reload = False  # si importamos los resultados ya guardados
    plotting = True

    stats.Analisis_Intensidad(n0s, n1s, plotting, reload, thetacs, corrections)

    print('Final!')
