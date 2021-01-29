from numpy import arange, ndenumerate, copy, save, load

dt = 0.1
dx = 0.01
dy = 0.01
k = 0.0001
C = 0.003
x_max = 1
y_max = 1
t_max = 300
u_max = 1
N = 10                                         # Number of modes for training data ICs
SenWei = 2
max_itr = 30
act_function = 'linear'
opt_method = 'Adam'
loss_function = 'mean_squared_error' 
equation = 'wave'                             # heat, nonlinear_heat, wave, heat_2D
model_type = 'Dense'                          # CNN, Dense, CNN_2D
IC = 2                                      # Choose from the following initial conditions
# f = lambda t: -t**2*(t-1)**2*(t-1/2)**2     # IC = 1
# f = lambda t: -t**2*(t-1)**2*(t+1/2)**2     # IC = 2    
# f = lambda t: -t**2*(t-1)**2*(t-1/4)**2     # IC = 3
# f = lambda t: -(t-1)**2*(2*t+1)**2          # IC = 4
# f = lambda t: -(t-1)**2*(t+1)*(2*t-1)**2    # IC = 5
# f_2D = lambda x, y: u_max*2**8*x**2*y**2*(x-1)**2*(y-1)**2 # IC = 6 for 2D model
# step function     # IC = 'step'



file_name = equation + '_Time{}'.format(t_max)+'_N{}_spline_reconstruction_SenWei{}_IC{}_dx{}_omega_0random_itr{}'.format(N,SenWei,IC,dx,max_itr)
x = arange(0,x_max+dx,dx)
y = arange(0,y_max+dy,dy)
t = arange(0,t_max+dt,dt)
cx = len(x)
cy = len(y)
r = len(t)
        