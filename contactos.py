import numpy as np

def calcular_contactos(dx, dy, cx, cy, idxx, idxy, theta):

       ''' Conocidas las coordendas (cx,cy) en el interior de la celda,
       podemos encontrar la distancia a las esquinas para determinar
       el próximo contacto.

       dx -- discretization in x axis
       dy -- discretization in y axis
       cx -- coordenada x
       cy -- coordenada y
       idxx -- índice x de la celda
       idxy -- índice y de la celda
       '''

       #Extremo superior-derecha
       xext1 = dx
       yext1 = dy
       angulo1 = np.arctan(abs(yext1-cy)/abs(xext1-cx))

       #Extremo superior-izquierda
       xext2 = 0.0
       yext2 = dy
       angulo2 = np.pi - np.arctan(abs(yext2-cy)/abs(xext2-cx))

       #Extremo inferior-izquierda
       xext3 = 0.0
       yext3 = 0.0
       angulo3 = np.pi + np.arctan(abs(yext3-cy)/abs(xext3-cx))

       #Extremo inferior-derecha
       xext4 = dx
       yext4 = 0.0
       angulo4 = 2*np.pi - np.arctan(abs(yext4-cy)/abs(xext4-cx))

       assert angulo1 < angulo2 < angulo3 < angulo4

       if 0 <= theta and theta < np.pi/2:

           if theta <= angulo1: #hacia la derecha
               ix, iy = idxx+1, idxy
               mov_type='r'
               #Las modificaciones sobre las coordenadas
               dx = dx - cx
               dy = np.tan(theta)*dx

           else:  #hacia arriba
               ix, iy = idxx, idxy+1
               mov_type='u'

               #Las modificaciones sobre las coordenadas
               dy = dy - cy
               dx = dy/np.tan(theta)


       elif np.pi/2 <= theta and theta < np.pi:

           if angulo1 < theta and theta <= angulo2: #hacia arriba
               ix, iy = idxx, idxy+1
               mov_type='u'

               dy = dy - cy
               dx = - dy*np.tan(theta - np.pi/2)
           else: #hacia izquierda
               ix, iy = idxx-1, idxy
               mov_type='l'

               dx = - cx
               dy =  dx*np.tan(theta - np.pi)

       elif np.pi <= theta and theta < 3*np.pi/2:

           if angulo2 < theta and theta <= angulo3: #hacia izquierda
               ix, iy = idxx-1, idxy
               mov_type='l'

               dx = - cx
               dy =  dx*np.tan(theta - np.pi)

           else: #hacia abajo
               ix, iy = idxx, idxy-1
               mov_type='d'

               dy =  - cy
               dx = dy*np.tan(3*np.pi/2 - theta)

       else:

           if angulo3 < theta  and theta<= angulo4: #hacia abajo
               ix, iy = idxx, idxy-1
               mov_type='d'

               dy =  - cy
               dx =  -dy*np.tan(theta + 3*np.pi/2)

           else: #hacia derecha
               ix, iy = idxx+1, idxy
               mov_type='r'
               dx = dx - cx
               dy = - dx/np.tan(theta - 3*np.pi/2)

       return dx, dy, ix, iy, mov_type


if __name__ == '__main__':

    calcular_contactos(0.2, 0.2, 0.1, 0.1, 1, 1, np.pi/3)


