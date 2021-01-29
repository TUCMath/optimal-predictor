# solver backward-time central-space method (implicit method)
# Boundary condition is u(0,t)=0 and u_x(1,t)=0
import numpy as np

def BTCS(dt,dx,t_max,x_max,k,u0):
    s = k*dt/dx**2
    x = np.arange(0,x_max+dx,dx) 
    t = np.arange(0,t_max+dt,dt)
    r = len(t)
    cx = len(x)

    A1 = np.diag([1+2*s]*(cx-1))
    A2 = np.diag([-s]*(cx-2),1)
    A3 = np.diag([-s]*(cx-2),-1)
    A = A1+A2+A3
    A[cx-2,cx-3] = -2*s

    

    u = np.zeros([r,cx])
    u[0,:] = u0
    u[:,0] = np.zeros(r)


    for n in range(1,r):
        u[n,1:cx] = np.linalg.solve(A,u[n-1,1:cx])

    return u