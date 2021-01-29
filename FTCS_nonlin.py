# Solver forward-time central-space method (explicit method)

import numpy as np

def FTCS_nonlin(dt,dx,t_max,x_max,k,u0):
    s = k*dt/dx**2
    x = np.arange(0,x_max+dx,dx) 
    t = np.arange(0,t_max+dt,dt)
    r = len(t)
    cx = len(x)
    u = np.zeros([r,cx])
    u[0,:] = u0
    for n in range(0,r-1):
        for j in range(1,cx-1):
            u[n+1,j] = u[n,j] + s*(u[n,j-1] - 2*u[n,j] + u[n,j+1]) - dt*(u[n,j])**3
        j = cx-1 
        u[n+1, j] = u[n,j] + s*(u[n,j-1] - 2*u[n,j] + u[n,j-1]) - dt*(u[n,j])**3
        j = 0
        u[n+1, j] = 0
    return u