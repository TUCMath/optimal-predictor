# Solver forward-time central-space method (explicit method) for 2D heat equation

import numpy as np

def FTCS_2D(dt,dx,dy,t_max,x_max,y_max,k,u0_2D):
    sx = k*dt/dx**2
    sy = k*dt/dy**2
    x = np.arange(0,x_max+dx,dx) 
    y = np.arange(0,y_max+dy,dy) 
    t = np.arange(0,t_max+dt,dt)
    r = len(t)
    cx = len(x)
    cy = len(y)
    u = np.zeros([r,cx,cy])
    u[0,:,:] = u0_2D
    for n in range(0,r-1):
        for i in range(1,cx-1):
            for j in range(1,cy-1):
                u[n+1,i,j] = u[n,i,j] + sx*(u[n,i-1,j] - 2*u[n,i,j] + u[n,i+1,j]) \
                            + sy*(u[n,i,j-1] - 2*u[n,i,j] + u[n,i,j+1]) 
            
            j = cy-1
            u[n+1,i,j] = u[n,i,j] + sx*(u[n,i-1,j] - 2*u[n,i,j] + u[n,i+1,j]) \
                            + sy*(u[n,i,j-1] - 2*u[n,i,j] + u[n,i,j-1]) 
            j= 0
            u[n+1,i,j] = u[n,i,j] + sx*(u[n,i-1,j] - 2*u[n,i,j] + u[n,i+1,j]) \
                            + sy*(u[n,i,j+1] - 2*u[n,i,j] + u[n,i,j+1]) 
        
        i = cx-1 
        for j in range(1,cy-1):
            u[n+1,i,j] = u[n,i,j] + sx*(u[n,i-1,j] - 2*u[n,i,j] + u[n,i-1,j]) \
                             + sy*(u[n,i,j-1] - 2*u[n,i,j] + u[n,i,j+1]) 
        
        u[n+1,cx-1,cy-1] = u[n,cx-1,cy-1] + sx*(u[n,cx-2,cy-1] - 2*u[n,cx-1,cy-1] + u[n,cx-2,cy-1]) \
                             + sy*(u[n,cx-1,cy-2] - 2*u[n,cx-1,cy-1] + u[n,cx-1,cy-2]) 
        
        u[n+1,cx-1,0] = u[n,cx-1,0] + sx*(u[n,cx-2,0] - 2*u[n,cx-1,0] + u[n,cx-2,0]) \
                             + sy*(u[n,cx-1,1] - 2*u[n,cx-1,0] + u[n,cx-1,1]) 
        
        i = 0
        for j in range(0,cy):
            u[n+1, i, j] = 0 

    return u