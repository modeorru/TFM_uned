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
    def __init__(self, num_rayos, Lx, Ly, nx, ny, As, folder):
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
        self.num_rayos = num_rayos
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.As = As
        self.folder = folder

    #########################################
    ######## ANÁLISIS ESTADÍSTICOS ##########
    #########################################

    def Analisis_Intensidad(self, plotting):
        estudios_estadisticos.analisis_intensidad_modulaciones_suaves(self.num_rayos, self.Lx, self.Ly,
                                                                      self.nx, self.ny, plotting,
                                                                      self.folder, self.As)


if __name__ == '__main__':

    # Definimos las variables a usar
    num_rayos = 1000
    Lx = 100
    Ly = 100
    nx = 100
    ny = 100
    As = np.linspace(0.1, 0.9, 9)
    folder = Path('results/modulaciones_suaves')
    folder.mkdir(exist_ok=True)

    # CREAMOS LA CLASE ESTADISTICAS
    stats = estadisticas(num_rayos, Lx, Ly, nx, ny, As, folder)

    # ANÁLISIS INTENSIDAD Y STD EN LA RED

    plotting = True

    stats.Analisis_Intensidad(plotting)
