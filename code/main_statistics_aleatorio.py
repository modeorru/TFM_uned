from __future__ import division
import numpy as np
import os
from tqdm import tqdm
import utilidades
import plot as pp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import estudios_estadisticos


class estadisticas():
    def __init__(self, samples, num_rayos, Lx, Ly, nx, ny, folder):
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
        self.folder = folder

    #########################################
    ######## ANÁLISIS ESTADÍSTICOS ##########
    #########################################

    def Analisis_Intensidad(self, n0s, n1s, plotting, reload):
        estudios_estadisticos.analisis_intensidad(self.samples, self.num_rayos, n0s, n1s, self.Lx, self.Ly,
                                                  self.nx, self.ny, plotting, reload, self.folder)

    def Analisis_Longitudes(self, n0s, n1s, reload):
        estudios_estadisticos.analisis_longitudes(self.num_rayos, n0s, n1s, self.Lx, self.Ly, self.nx,
                                                  self.ny, self.folder, reload)

    def Analisis_Correlacion_Longitud(self, n0s, n1s, reload):
        estudios_estadisticos.analisis_correlacion_longitud(self.num_rayos, n0s, n1s, self.Lx, self.Ly, self.nx,
                                                            self.ny, self.folder, reload)

    def Analisis_Punto_Fijado(self, n0, n1):
        estudios_estadisticos.analisis_punto_fijado(self.samples, n0, n1, self.nx,
                                                    self.ny, self.folder)

    def Analisis_Punto_Fijado_para_todo_n(self, n0, n1):
        estudios_estadisticos.analisis_punto_fijado_para_todo_n(self.samples, n0, n1, self.nx,
                                                                self.ny, self.folder)

    def Analisis_Direccion_Fija_Diferente_Ruido(self, n0s, n1s, chosen_theta):
        estudios_estadisticos.analisis_direccion_fija_diferente_ruido(self.samples, n0s, n1s, self.nx, self.ny,
                                                                      self.Lx, self.Ly, self.folder,
                                                                      chosen_theta)

    def Analisis_Ruido_Fijo_Diferente_Direccion(self, n0s, n1s, chosen_theta):
        estudios_estadisticos.analisis_ruido_fijo_diferente_direccion(self.samples, n0s, n1s, self.nx, self.ny,
                                                                      self.Lx, self.Ly, self.folder, chosen_theta)

    def Analisis_Tiempo_Minimizante(self, n0s, n1s, chosen_theta):
        estudios_estadisticos.analisis_tiempo_minimizante(self.samples, n0s, n1s, self.nx, self.ny, self.Lx, self.Ly,
                                                          self.folder, chosen_theta)

    def Analisis_Cruces(self, n0s, n1s, reload):
        estudios_estadisticos.analisis_cruces(reload, self.samples, self.num_rayos, n0s, n1s, self.nx, self.ny, self.Lx, self.Ly,
                                              self.folder)


if __name__ == '__main__':

    # Definimos las variables a usar
    samples = 100
    num_rayos = 1000
    Lx = 100
    Ly = 100
    nx = 100
    ny = 100
    folder = Path('results/aleatorio')
    folder.mkdir(exist_ok=True)

    # CREAMOS LA CLASE ESTADISTICAS
    stats = estadisticas(samples, num_rayos, Lx, Ly, nx, ny, folder)

    # ANÁLISIS INTENSIDAD Y STD EN LA RED

    n1s = np.arange(1.1, 2.0, 0.1)
    n0s = np.ones(len(n1s))*1
    reload = False  # si importamos los resultados ya guardados
    plotting = False

    stats.Analisis_Intensidad(n0s, n1s, plotting, reload)

    # ANÁLISIS DE LA INTENSIDAD EN UNA DIRECCIÓN FIJA
    chosen_theta_list = np.linspace(0, 45, 4, endpoint=True)*np.pi/180
    # Escaneamos para una theta fija
    for chosen_theta in chosen_theta_list:
        stats.Analisis_Direccion_Fija_Diferente_Ruido(n0s, n1s, chosen_theta)

    # ANÁLISIS DE LA INTENSIDAD PARA VARIAS DIRECCIONES Y RUIDO FIJO
    chosen_theta = np.linspace(start=0, stop=np.pi/2, num=10)
    # Escaneamos para un par de índices fijos
    for n0, n1 in zip(n0s, n1s):
        stats.Analisis_Ruido_Fijo_Diferente_Direccion(n0, n1, chosen_theta)

    # ANÁLISIS DEL TIEMPO MINIMIZANTE EN DIRECCIÓN FIJA, DIFERENTE RUIDO
    chosen_theta_list = np.linspace(start=0, stop=np.pi/4, num=4)
    for chosen_theta in chosen_theta_list:
        stats.Analisis_Tiempo_Minimizante(n0s, n1s, chosen_theta)

    # ANÁLISIS DE RELACIÓN LONGITUD RAYO CON DISTANCIA REAL
    stats.Analisis_Longitudes(n0s, n1s, reload=False)

    # ANÁLISIS DE RELACIÓN LONGITUD RAYO CON DISTANCIA REAL EN UN EJE CONCRETO
    stats.Analisis_Correlacion_Longitud(n0s, n1s, reload=False)

    # ANÁLISIS DE DISTRIBUCIÓN DE CRUCES
    stats.Analisis_Cruces(n0s, n1s, reload=False)

    # ANÁLISIS DEL HISTOGRAMA DE INTENSIDAD PARA DIFERENTES PUNTOS DE LA RED
    stats.Analisis_Punto_Fijado_para_todo_n(n0s[::2], n1s[::2])
    for n0, n1 in zip(n0s, n1s):
        n0 = np.around(n0, 1)
        n1 = np.around(n1, 1)
        stats.Analisis_Punto_Fijado(n0, n1)

    print('Final!')




