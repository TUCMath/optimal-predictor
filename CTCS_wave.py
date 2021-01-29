# Solver central-time central-space method for wave equation

import numpy as np

def CTCS_wave(dt,dx,t_max,x_max,C,u0,Du0):
    s = C*dt/dx
    x = np.arange(0,x_max+dx,dx) 
    t = np.arange(0,t_max+dt,dt)
    r = len(t)
    c = len(x)
    u = np.zeros([r,c])
    u[0,:] = u0
    u[1,:] = u0+dt*Du0
    for n in range(1,r-1):
        for j in range(1,c-1):
            u[n+1,j] = - u[n-1,j] + 2*u[n,j] + s**2*(u[n,j-1] - 2*u[n,j] + u[n,j+1])
        j = c-1 
        u[n+1, j] = 0         # BC1
        j = 0
        u[n+1, j] = 0         # BC1
    return u