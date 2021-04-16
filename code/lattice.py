import numpy as np
from tqdm import tqdm
import utilidades


class lattice():

    '''
    Clase que describe la red cuadrada sobre la cual se realiza el estudio de
    la propagación de los rayos.
    '''

    def __init__(self, num_rayos, Lx, Ly, nx, ny, n0, n1, loc_eje=None, A=None, caos=None,
                 random_theta=False, chosen_theta=None):
        '''
        Argumentos importantes:

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
            -> n0: float
                    límite inferior para el índice de refracción.
            -> n1: float
                    límite superior para el índice de refracción.
            -> loc_eje: int o None
                    if None: no se introduce un eje de cambio de índice.
                    if int: se introduce un eje que separa dos índices.
            -> error: bool
                    se hace True si ha aparecido algún error durante la simulación.
            -> A: float < 1 or None
                    se introducen modulaciones suaves en el índice de refracción de
                    acuerdo a la ec.(1)
                            n_{ij} =1 + A·sin(2·pi·i/L)·sin(2·pi·j/L)       (1)
            -> caos: 1 or None
                    if 1: el índice de refracción obedecerá ec.(2)
                                n = U[n0,n1]                                (2)
            -> random_theta: bool
                    if True: se eligen direcciones random de los rayos.
                    else: los rayos se equiespacian en [0,2·pi]
            -> chosen_theta: float or None
                    si se especifica, todos los rayos obedecen este ángulo de salida.

        '''

        self.error = False
        self.num_rayos = num_rayos  # número de rayos
        self.Lx = Lx  # longitud eje x
        self.Ly = Ly  # longitud eje y
        self.nx = nx  # número de celdas x
        self.ny = ny  # número de celdas y

        self.intensities = np.zeros((self.nx, self.ny))

        self.n0 = n0  # índice ref. 1
        self.n1 = n1  # índice ref. 2

        self.dx = self.Lx/self.nx  # discretización eje x
        self.dy = self.Ly/self.ny  # discretización eje y

        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        self.xv, self.yv = np.meshgrid(x, y, sparse=False, indexing='xy')

        # Asignar índices de refracción y condiciones iniciales
        self.inicializar(loc_eje=loc_eje, A=A, caos=caos,
                         random_theta=random_theta, chosen_theta=chosen_theta)

    def inicializar(self, loc_eje=None, A=None, caos=None, random_theta=False, chosen_theta=None):

        # ASIGNACIÓN DE LOS ÍNDICES DE REFRACCIÓN

        self.n = utilidades.asignar(self.nx, self.ny, self.n0, self.n1, loc_eje, A, caos)

        # INICIALIZACIÓN DE ÍNDICES Y COORDENDAS DE LOS RAYOS

        # Índice de la celda donde empiezan los rayos
        idxx = [int(self.nx/2) for i in range(self.num_rayos)]
        idxy = [int(self.ny/2) for i in range(self.num_rayos)]

        # Coordenadas
        #c0x = [np.random.rand()*self.dx for i in range(num_rayos)]
        #c0y = [np.random.rand()*self.dy for i in range(num_rayos)]
        c0x = [self.dx/2 for i in range(self.num_rayos)]
        c0y = [self.dy/2 for i in range(self.num_rayos)]

        self.idxx_rayos = []
        self.idxy_rayos = []
        self.coordx_rayos = []
        self.coordy_rayos = []

        for i in range(self.num_rayos):
            self.coordx_rayos.append([c0x[i]])
            self.coordy_rayos.append([c0y[i]])
            self.idxx_rayos.append([idxx[i]])
            self.idxy_rayos.append([idxy[i]])

        self.movimiento = np.ones(self.num_rayos)  # se sigue propagando el rayo?

        if random_theta:
            theta = np.random.rand(self.num_rayos, 1)*2*np.pi  # direccion del rayo random (0,2*pi)
        elif chosen_theta is not None:
            # solo valido cuando se tiene 1 rayo
            theta = np.array([chosen_theta for i in range(self.num_rayos)])
            theta = theta.reshape((self.num_rayos, 1))
        else:
            theta = np.linspace(0, 2*np.pi, self.num_rayos)
            theta = theta.reshape((self.num_rayos, 1))

        self.theta = theta.tolist()

    def actualizacion(self, i, idxx, idxy, cx, cy, dx, dy, ix, iy, theta1, mov_type):
        ''' Nuevo ángulo por Ley Snell. Ángulos en rad
            idxx -- idx x the la celda actual
            idxy -- idx y the la celda actual
            cx -- coordenada x en esa celda
            cy -- coordenada y en esa celda
            ix -- idx x the la nueva celda
            ix -- idx y the la nueva celda
            th -- ángulo respecto a la horizontal
            mov_type -- hacia q lado se mueve
        '''
        tol = 10**(-5)  # tolerance value to avoid infinities

        # Actualizamos las posiciones y rotamos ángulo
        # Necesitamos realizar una translación del eje para tener
        # el perpendicular con el eje de la nueva celda

        theta1_orig = theta1
        cx_orig = cx
        cy_orig = cy

        self.intensities[idxx, idxy] += (np.sqrt(dx**2 + dy**2))

        # Actualizamos las intensidades por celda
        # SI EL NUEVO ÍNDICE ESTÁ DENTRO DE LA CUADRÍCULA
        if 0 <= ix <= self.nx-1 and 0 <= iy <= self.ny-1:

            if mov_type == 'r':
                cx = 0
                cy += dy
                if 2*np.pi/3 <= theta1 and theta1 <= 2*np.pi:
                    theta1 = 2*np.pi - theta1

            if mov_type == 'u':
                cy = 0
                cx += dx
                if theta1 < np.pi:
                    theta1 -= np.pi/2

            if mov_type == 'l':
                cx = self.dx
                cy += dy
                theta1 -= np.pi

            if mov_type == 'd':
                cy = self.dy
                cx += dx
                theta1 = 3*np.pi/2 - theta1

            if dx != 0:
                if abs(dx) == dx:
                    cx += tol
                else:
                    cx -= tol
            if dy != 0:
                if abs(dy) == dy:
                    cy += tol
                else:
                    cy -= tol

            theta1 = abs(theta1)

            assert theta1 < np.pi/2, 'En este sistema referencia siempre menor que pi/2'

            # Pasamos a buscar nuevo ángulo
            n1 = self.n[idxy, idxx]  # índice en celda previa
            n2 = self.n[iy, ix]  # índice en celda nueva
            nmin = min(n1, n2)
            nmax = max(n1, n2)
            thetac = np.arcsin(nmin/nmax)  # ángulo crítico

            if theta1 < thetac or n2 > n1:  # no reflexión total

                theta2 = np.arcsin((n1/n2)*np.sin(theta1))

                if mov_type == 'r':
                    if theta1_orig > np.pi:
                        theta2 = 2*np.pi - theta2
                elif mov_type == 'u':
                    if theta1_orig < np.pi/2:
                        theta2 = np.pi/2 - theta2
                    else:
                        theta2 = np.pi/2 + theta2
                elif mov_type == 'l':
                    if theta1_orig < np.pi:
                        theta2 = np.pi - theta2
                    else:
                        theta2 = np.pi + theta2
                else:  # down
                    if theta1_orig < 3*np.pi/2:
                        theta2 = 3*np.pi/2 - theta2
                    else:
                        theta2 = 3*np.pi/2 + theta2
            else:  # reflexión total
                if mov_type == 'u':
                    if theta1_orig < np.pi/2:
                        theta2 = 3*np.pi/2 + theta1
                    else:
                        theta2 = 3*np.pi/2 - theta1
                if mov_type == 'l':
                    if theta1_orig < np.pi:
                        theta2 = theta1
                    else:
                        theta2 = 2*np.pi - theta1
                if mov_type == 'd':
                    if theta1_orig < 3*np.pi/2:
                        theta2 = np.pi/2 + theta1
                    else:
                        theta2 = np.pi/2 - theta1
                if mov_type == 'r':
                    if theta1_orig < np.pi/2:
                        theta2 = np.pi - theta1
                    else:
                        theta2 = np.pi + theta1

                ix = idxx
                iy = idxy
                cx = cx_orig + dx - tol
                cy = cy_orig + dy - tol

            self.theta[i].append(theta2)

        else:
            cx += dx
            cy += dy

            if abs(dx) == dx:
                cx -= tol
            else:
                cx += tol
            if abs(dy) == dy:
                cy -= tol
            else:
                cy += tol
            ix = idxx
            iy = idxy

            self.movimiento[i] = 0

        # Actualizamos posiciones e índices
        #print('Recuerda estas', cx, cy, ix, iy)
        self.coordx_rayos[i].append(cx)
        self.coordy_rayos[i].append(cy)
        self.idxx_rayos[i].append(ix)
        self.idxy_rayos[i].append(iy)

    def evolucion(self):

        for i in tqdm(range(self.num_rayos)):

            t = 0

            while self.movimiento[i] != 0:  # si no ha llegado a los extremos del cubo

                idxx, idxy = self.idxx_rayos[i][-1], self.idxy_rayos[i][-1]  # en que celda están

                # que coordenadas dentro de la celda
                cx, cy = self.coordx_rayos[i][-1], self.coordy_rayos[i][-1]
                th = self.theta[i][-1]  # que ángulo respecto la horizontal

                # CALCULAMOS CONTACTOS
                dx, dy, ix, iy, mov_type = utilidades.calcular_contactos(
                    self.dx, self.dy, cx, cy, idxx, idxy, th)

                # ACTUALIZAMOS POSICIONES

                self.actualizacion(i, idxx, idxy, cx, cy, dx, dy, ix, iy, th, mov_type)
                t += 1
                if t > 5000:  # en caso que el rayo tarde mucho se descarta
                    print('Too long, discard that example')
                    self.movimiento[i] = 0
                    self.error = True
