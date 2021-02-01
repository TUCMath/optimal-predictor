from numpy import load, array, zeros, reshape, cos, pi, arange, append, ones, random
from numpy import matmul, diag, piecewise, sum, ravel, save, meshgrid
from numpy.linalg import norm
from numpy.random import rand
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Lambda, Reshape, Flatten, SimpleRNN
from keras import backend as K
from keras.losses import mean_absolute_error
from parameters import *
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from FTCS import *
from FTCS_nonlin import *
from scipy.optimize import Bounds, minimize, BFGS, SR1, fminbound
from generate_data import *
from generate_model import *
from scipy import trapz 
from scipy.interpolate import CubicSpline, interp2d, griddata
from nodes import *

training_data = generate_data(equation)
model = generate_model(model_type)

input = training_data[0]
output = training_data[1]

model.fit(input, output, batch_size=1000, epochs=4, verbose=0)

grads = K.gradients(model.output, model.input)[0]
gradient = K.function(model.input, grads)

All_ICs = [0, lambda t: -t**2*(t-1)**2*(t-1/2)**2,               # IC1
            lambda t: -t**2*(t-1)**2*(t+1/2)**2,                 # IC2    
            lambda t: -t**2*(t-1)**2*(t-1/4)**2,                 # IC3
            lambda t: -(t-1)**2*(2*t+1)**2,                      # IC4
            lambda t: -(t-1)**2*(t+1)*(2*t-1)**2,                # IC5
            lambda x, y: u_max*2**8*x**2*y**2*(x-1)**2*(y-1)**2] # IC6

if '2D' in equation:
    f_2D = All_ICs[IC]
    u0_2D = array([[f_2D(i,j) for i in x] for j in y])
else:
    if IC == 'step':
        u0 = u_max*piecewise(x,[x < 0.5, x >= 0.5], [0 ,1]) # step function
    else:
        f = All_ICs[IC]
        x0 = fminbound(f, 0, 1)
        f_max = -f(x0)
        u0 = -u_max/f_max*f(x)
    Du0 = zeros(cx)

if equation == 'heat':
    u_real = FTCS(dt, dx, t_max, x_max, k, u0)
if equation == 'nonlinear_heat':
    u_real = FTCS_nonlin(dt, dx, t_max, x_max, k, u0)
if equation == 'wave':
    u_real = CTCS_wave(dt, dx, t_max, x_max, C, u0, Du0)
if equation == 'heat_2D':
    u_real = FTCS_2D(dt, dx, dy, t_max, x_max, y_max, k, u0_2D)

if '2D' in equation:
    omega_0 = rand(cx*cy)
    bounds = Bounds(zeros((cx*cy)), ones((cx*cy)))
else:
    omega_0 = rand(cx)
    #omega_0 = 5*[1] + 20*[0] + 10*[0] + 20*[0] + (cx-65)*[1] + 10*[0]
    bounds = Bounds([0]*cx , [1]*cx)


def reconstruct(u_real,omega):
    if '2D' in equation:
        omega = omega.reshape((cx,cy))
        x_id = nodes_2D(omega)[0]
        y_id = nodes_2D(omega)[1]

        if not x_id and not y_id:

            u_recon = zeros((r,cx,cy))

            return u_recon
        else:

            x_id = x_id + [0]*cx
            y_id = y_id + list(range(0,cy))
            x_sen = x[x_id]
            y_sen = y[y_id]
            points = np.column_stack((x_sen,y_sen))

            x_grid, y_grid = meshgrid(x, y)
            
            u_recon = zeros((r,cx,cy))
            for i in range(0,r):
                u_sen = u_real[i,x_id,y_id]
                # sp = interp2d(x_sen, y_sen, u_sen, kind='cubic')
                sp = griddata(points, u_sen, (x_grid, y_grid), method='cubic', fill_value=0)
                u_recon[i,:,:] = sp
            
            return u_recon


    else:    
        id = nodes(omega)

        if not id:

            u_recon = zeros((r,cx))

            return u_recon
            
        else:
            if id[0] != 0:
                id.insert(0,0)

            u_sen = u_real[:,id]
            x_sen = x[id]

            u_recon = zeros((r,cx))
            for i in range(0,r):
                sp = CubicSpline(x_sen,u_sen[i,:])
                u_recon[i,:] = sp(x)
    
        return u_recon 


def LQ_cost(omega):

    u_recon = reconstruct(u_real,omega)
    u_pred = model.predict(u_recon)

    cost = (u_pred - u_real)**2
    
    if '2D' in equation:
        cost = trapz(trapz(trapz(cost, dx=dx), dx=dy), dx=dt) 
    else:
        cost = trapz(trapz(cost, dx=dx), dx=dt) 
    
    cost += SenWei*sum(omega)
    
    return cost


def LQ_grad(omega):

    u_recon = reconstruct(u_real,omega)
    u_pred = model.predict(u_recon)
    u_prime = gradient(u_recon)

    D = 2*u_prime*(u_pred-u_real)
    
    if '2D' in equation:
        grad = zeros((cx,cy))

        for i in range(cx):
            for j in range(cy):
                grad[i,j] = trapz(D[:,i,j], dx=dt) + SenWei
    
    else:
        grad = zeros(cx)

        for i in range(cx):
            grad[i] = trapz(D[:,i], dx=dt) + SenWei
    
    return grad.ravel()

def callback_cost(omega,state):
    cost_values.append(state.fun)

cost_values = []

res = minimize(LQ_cost,omega_0,method='trust-constr',
           jac=LQ_grad,
           bounds=bounds,
           options={'verbose': 1, 'maxiter': max_itr, 'disp': True},
           callback=callback_cost)


if '2D' in equation:
    omega_b = res.x
    u_pred = model.predict(reconstruct(u_real,omega_b))
    omega_b = omega_b.reshape((cx,cy)) 
else:
    omega_ = [i for i, value in enumerate(res.x) if value < 0.5]
    omega_b = ones(cx)
    omega_b[omega_] = 0
    u_pred = model.predict(reconstruct(u_real,omega_b))

save('./data/u_real_'+file_name,u_real)
save('./data/u_pred_'+file_name,u_pred)
save('./data/omega_'+file_name,omega_b)
save('./data/cost_values_'+file_name,cost_values)
