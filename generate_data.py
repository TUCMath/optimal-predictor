from numpy import zeros, sin, cos, pi, save
import matplotlib.pyplot as plt
from parameters import *
from FTCS import *
from FTCS_nonlin import *
from CTCS_wave import *
from FTCS_2D import *

def generate_data(equation):
    if '2D' in equation:
        m = N**2*(r-1)
        input = zeros((m,cx,cy))
        output = zeros((m,cx,cy))

        u0 = zeros((cx,cy))
        def IC_2D(x,y,alpha,beta,u_max):
            u = u_max*cos(alpha*pi*x)*cos(beta*pi*y)
            return u

        n = 0
        for alpha in range(0,N):
            for beta in range(0,N):
                u0_2D = IC_2D(x,y,alpha,beta,u_max)
                
                if equation == 'heat_2D':
                    u = FTCS_2D(dt,dx,dy,t_max,x_max,y_max,k,u0_2D)
                
                for s in range(0,r-1):
                    input[n,:,:] = u[s,:,:]
                    output[n,:,:] = u[s+1,:,:]
                    n = n+1


    else:   
        m = N*(r-1)
        input = zeros((m,cx))
        output = zeros((m,cx))

        u0 = zeros(cx)
        def IC(x,alpha,u_max):
            u = u_max*cos(alpha*pi*x)
            return u

        n=0
        for alpha in range(0,N):
            u0 = IC(x,alpha,u_max)
            Du0 = zeros(cx)
            # plt.plot(x, u0)

            if equation == 'heat':
                u = FTCS(dt, dx, t_max, x_max, k, u0)
            if equation == 'nonlinear_heat':
                u = FTCS_nonlin(dt, dx, t_max, x_max, k, u0)
            if equation == 'wave':
                u = CTCS_wave(dt, dx, t_max, x_max, C, u0, Du0)
            
            for s in range(0,r-1):
                input[n,:] = u[s,:]
                output[n,:] = u[s+1,:]
                n = n+1
        
        # plt.xlabel('x')
        # plt.ylabel('u0(x)')
        # plt.xlim(0,1)
        # plt.title('Training Initial Conditions')
        # plt.savefig('./figs/Dense-training_ICs-%dmodes.png'%N)
    
    save('./training-data/training_inputs_'+equation+'_%dmodes.npy'%N, input)
    save('./training-data/training_outputs_'+equation+'_%dmodes.npy'%N, output)
    return [input,output]