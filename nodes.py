from parameters import *

def nodes(omega):
    return [i for i, value in enumerate(omega) if value >= 0.5]

def nodes_2D(omega):
    x_sensor = []
    y_sensor = []
    omega_sensor = copy(omega)
    for (i,j), value in ndenumerate(omega):
        if value >=0.5:
            x_sensor.append(i)
            y_sensor.append(j)
            omega_sensor[(i,j)] = -2*u_max
        else:
            omega_sensor[(i,j)] = None
    return [x_sensor, y_sensor, omega_sensor]