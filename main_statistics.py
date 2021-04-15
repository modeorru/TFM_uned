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


def main(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje, A, caos, folder, name, statistics, samples):

    print('Simulando', num_rayos, 'en', samples, 'samples')


if __name__ == '__main__':

    # Extaremos las variables de la función read
    num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje, A, caos, folder, name, statistics, samples = utilidades.read()
    main(num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje, A, caos, folder, name, statistics, samples)
