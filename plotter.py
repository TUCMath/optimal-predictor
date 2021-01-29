from parameters import *
from animate_2D import *
from animate import *
from os import remove
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
import tikzplotlib
import matplotlib.pyplot as plt
from numpy import linalg
from nodes import *
from scipy import trapz 
from matplotlib.offsetbox import AnchoredText

font = {'family' : 'serif',
        'size'   : 12}

matplotlib.rc('font', **font)
## To plot cost #################################################################################
u_real = load('./data/u_real_'+file_name+'.npy')
u_pred = load('./data/u_pred_'+file_name+'.npy')
omega_b = load('./data/omega_'+file_name+'.npy')
cost_values = load('./data/cost_values_'+file_name+'.npy')
if '2D' in equation:
    plt.figure()
    for i in range(0,4):
        SenWei = [5,10,15,20][i]
        file_name = equation + '_Time{}'.format(t_max) \
            +'_N{}_spline_reconstruction_SenWei{}_IC{}_dx{}_omega_0random_itr{}'.format(N,SenWei,IC,dx,max_itr)
        cost_values = load('./data/cost_values_'+file_name+'.npy')
        plt.plot(cost_values,linewidth=2)
    plt.legend([r'$\alpha = 5$',r'$\alpha = 10$',r'$\alpha = 15$',r'$\alpha = 20$'])
    plt.xlabel('Iterations')
    plt.ylabel(r'$J(\omega)$')
    plt.xlim(1,max_itr)
    plt.savefig('./figs/cost_'+file_name+'.png',bbox_inches='tight')
    plt.savefig('./figs/cost_'+file_name+'.pdf',bbox_inches='tight')
    figpath = './figs/cost_'+file_name+'.tex'
    tikzplotlib.save(figure='gcf', filepath=figpath)
if 'IC1' in file_name and 'heat' in file_name:
    plt.figure()
    for i in ['IC1','IC2','IC3','ICstep']:
        new_file_name = file_name.replace('IC1',i)
        cost_values = load('./data/cost_values_'+new_file_name+'.npy')
        plt.plot(cost_values,linewidth=2)
    plt.legend([r'$u_0=x^2(x-1)^2(x-1/2)^2$',r'$u_0=x^2(x-1)^2(x+1/2)^2$',
                r'$u_0=x^2(x-1)^2(x-1/4)^2$',r'$H(x-1/2)$'])
    plt.xlabel('Iterations')
    plt.ylabel(r'$J(\omega)$')
    plt.xlim(1,max_itr)
    new_file_name = file_name.replace('IC1','IC123step')
    plt.savefig('./figs/cost_'+new_file_name+'.png')
    plt.savefig('./figs/cost_'+new_file_name+'.pdf')
    figpath = './figs/cost_'+new_file_name+'.tex'
    tikzplotlib.save(figure='gcf', filepath=figpath)

if 'IC1' in file_name and 'wave' in file_name:
    plt.figure()
    for i in ['IC1','IC2','IC3']:
        new_file_name = file_name.replace('IC1',i)
        cost_values = load('./data/cost_values_'+new_file_name+'.npy')
        plt.plot(cost_values,linewidth=2)
    plt.legend([r'$u_0=x^2(x-1)^2(x-1/2)^2$',r'$u_0=x^2(x-1)^2(x+1/2)^2$',
                r'$u_0=x^2(x-1)^2(x-1/4)^2$'])
    plt.xlabel('Iterations')
    plt.ylabel(r'$J(\omega)$')
    plt.xlim(1,max_itr)
    new_file_name = file_name.replace('IC1','IC123step')
    plt.savefig('./figs/cost_'+new_file_name+'.png')
    plt.savefig('./figs/cost_'+new_file_name+'.pdf')
    figpath = './figs/cost_'+new_file_name+'.tex'
    tikzplotlib.save(figure='gcf', filepath=figpath)

## To plot norm of error #########################################################################
u_real = load('./data/u_real_'+file_name+'.npy')
u_pred = load('./data/u_pred_'+file_name+'.npy')
omega_b = load('./data/omega_'+file_name+'.npy')
cost_values = load('./data/cost_values_'+file_name+'.npy')
if '2D' in equation:
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(0,4):
        SenWei = [5,10,15,20][i]
        file_name = equation + '_Time{}'.format(t_max) \
            +'_N{}_spline_reconstruction_SenWei{}_IC{}_dx{}_omega_0random_itr{}'.format(N,SenWei,IC,dx,max_itr)
        u_real = load('./data/u_real_'+file_name+'.npy')
        u_pred = load('./data/u_pred_'+file_name+'.npy')
        error = u_real - u_pred
        error_1norm = dx*dy*linalg.norm(error, ord=1, axis=(1,2))
        ax.plot(t, error_1norm, linewidth=2)
    ax.legend([r'$\alpha = 5$',r'$\alpha = 10$',r'$\alpha = 15$',r'$\alpha = 20$'])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\|u_r-u_p\|_1$')
    ax.set_xlim(0,t_max)
    fig.savefig('./figs/error_l1norm_'+file_name+'.png',bbox_inches='tight')
    fig.savefig('./figs/error_l1norm_'+file_name+'.pdf',bbox_inches='tight')
    figpath = './figs/error_l1norm_'+file_name+'.tex'
    tikzplotlib.save(figure='gcf', filepath=figpath)

if 'IC1' in file_name and 'heat' in file_name:
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    for i in ['IC1','IC2','IC3','ICstep']:
        new_file_name = file_name.replace('IC1',i)
        u_real = load('./data/u_real_'+new_file_name+'.npy')
        u_pred = load('./data/u_pred_'+new_file_name+'.npy')
        error = u_real - u_pred
        error_1norm = dx*linalg.norm(error, ord=1, axis=1)
        error_maxnorm = linalg.norm(error, ord=np.inf, axis=1)
        ax1.plot(t,error_1norm,linewidth=2)
        ax1 = fig1.add_subplot()
        ax2.plot(t,error_maxnorm,linewidth=2)
        ax2 = fig2.add_subplot()

    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$\|u_r-u_p\|_1$')
    ax1.set_xlim(0,t_max)
    ax1.legend([r'$u_0=x^2(x-1)^2(x-1/2)^2$',r'$u_0=x^2(x-1)^2(x+1/2)^2$',
                    r'$u_0=x^2(x-1)^2(x-1/4)^2$',r'$H(x-1/2)$'])
    fig1.savefig('./figs/error_l1norm_'+file_name+'.png')
    fig1.savefig('./figs/error_l1norm_'+file_name+'.pdf')

    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$\|u_r-u_p\|_\infty$')
    ax2.set_xlim(0,t_max)
    ax2.legend([r'$u_0=x^2(x-1)^2(x-1/2)^2$',r'$u_0=x^2(x-1)^2(x+1/2)^2$',
                    r'$u_0=x^2(x-1)^2(x-1/4)^2$',r'$H(x-1/2)$'])
    fig2.savefig('./figs/error_linftynorm_'+file_name+'.png')
    fig2.savefig('./figs/error_linftynorm_'+file_name+'.pdf')

if 'IC1' in file_name and 'wave' in file_name:
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    for i in ['IC1','IC2','IC3']:
        new_file_name = file_name.replace('IC1',i)
        u_real = load('./data/u_real_'+new_file_name+'.npy')
        u_pred = load('./data/u_pred_'+new_file_name+'.npy')
        error = u_real - u_pred
        error_1norm = dx*linalg.norm(error, ord=1, axis=1)
        error_maxnorm = linalg.norm(error, ord=np.inf, axis=1)
        ax1.plot(t,error_1norm,linewidth=2)
        ax1 = fig1.add_subplot()
        ax2.plot(t,error_maxnorm,linewidth=2)
        ax2 = fig2.add_subplot()

    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$\|u_r-u_p\|_1$')
    ax1.set_xlim(0,t_max)
    ax1.legend([r'$u_0=x^2(x-1)^2(x-1/2)^2$',r'$u_0=x^2(x-1)^2(x+1/2)^2$',
                    r'$u_0=x^2(x-1)^2(x-1/4)^2$'], loc='upper right')
    fig1.savefig('./figs/error_l1norm_'+file_name+'.png')
    fig1.savefig('./figs/error_l1norm_'+file_name+'.pdf')

    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$\|u_r-u_p\|_\infty$')
    ax2.set_xlim(0,t_max)
    ax2.legend([r'$u_0=x^2(x-1)^2(x-1/2)^2$',r'$u_0=x^2(x-1)^2(x+1/2)^2$',
                    r'$u_0=x^2(x-1)^2(x-1/4)^2$'], loc='upper right')
    fig2.savefig('./figs/error_linftynorm_'+file_name+'.png')
    fig2.savefig('./figs/error_linftynorm_'+file_name+'.pdf')

## To plot snapshots #############################################################################
u_real = load('./data/u_real_'+file_name+'.npy')
u_pred = load('./data/u_pred_'+file_name+'.npy')
omega_b = load('./data/omega_'+file_name+'.npy')
cost_values = load('./data/cost_values_'+file_name+'.npy')
error = u_real - u_pred
if '2D' in equation:
    x_grid, y_grid = meshgrid(x, y)
    x_sensor = nodes_2D(omega_b)[0]
    y_sensor = nodes_2D(omega_b)[1]
    #gs1 = gridspec.GridSpec(1, 1)
    #gs1.update(wspace=0.1, hspace=0.2)
    for i in range(0,4):
        s = i * t_max / 3
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x_grid,y_grid,error[round(s/dt),:,:],rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        ax.set_xlim(0,1)
        ax.set_xlabel(r'$x$')
        ax.set_ylim(0,1)
        ax.set_ylabel(r'$y$')
        ax.set_zlim(-u_max,u_max)
        ax.set_zlabel('Error')
        at = AnchoredText(r'$t=$%d'%s,
                  prop=dict(size=10), frameon=False,
                  loc='upper right',
                  )
        ax.add_artist(at)
        plt.savefig('./figs/snapshots{}_'.format(i)+file_name+'.pdf')
    fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    for i in range(0,4):
        SenWei = [5,10,15,20][i]
        file_name = equation + '_Time{}'.format(t_max) \
            +'_N{}_spline_reconstruction_SenWei{}_IC{}_dx{}_omega_0random_itr{}'.format(N,SenWei,IC,dx,max_itr)
        omega_b = load('./data/omega_'+file_name+'.npy')
        x_sensor = nodes_2D(omega_b)[0]
        y_sensor = nodes_2D(omega_b)[1]
        axsr = axs.reshape(4,)
        axsr[i].plot(x[x_sensor],y[y_sensor],'gs')
        axsr[i].set_xlim(0,1)
        axsr[i].set_xlabel(r'$x$')
        axsr[i].set_ylim(0,1)
        axsr[i].set_ylabel(r'$y$')
        axsr[i].set_title(r'$\alpha = {}$'.format(SenWei))
        axsr[i].grid(True)
    fig.savefig('./figs/sensor_shape_'.format(i)+file_name+'.pdf')
    fig.savefig('./figs/sensor_shape_'.format(i)+file_name+'.png')
if 'heat' in equation and '2D' not in equation:
    fig, axs = plt.subplots(4, sharex=True, sharey=True)
    x_sensor = x[nodes(omega_b)]
    y_sensor = array([0]*len(x_sensor))
    for i in range(0,4):
        s = i * t_max / 3
        axs[i].plot(x,u_real[round(s/dt),:])
        axs[i].plot(x,u_pred[round(s/dt),:],'--')
        axs[i].set_xlim(0,1)
        axs[i].set_ylim(0,u_max)
        at = AnchoredText(r'$t=$%d'%s,
                  prop=dict(size=10), frameon=False,
                  loc='upper right',
                  )
        axs[i].add_artist(at)
        axs[i].plot(x_sensor,y_sensor,'s')
    fig.text(0.5, 0.04, r'$x$', ha='center')
    fig.text(0.04, 0.5, r'$u_r(x,t)$ and $u_p(x,t)$', va='center', rotation='vertical')
    fig.savefig('./figs/snapshots_' + file_name + '.png')
    plt.savefig('./figs/snapshots_'+file_name+'.pdf')
    figpath = './figs/snapshots_'+file_name+'.tex'
    #tikzplotlib.save(figure='gcf', filepath=figpath)
if 'wave' in equation:
    fig, axs = plt.subplots(4, sharex=True, sharey=True)
    x_sensor = x[nodes(omega_b)]
    y_sensor = array([-u_max]*len(x_sensor))
    for i in range(0,4):
        s = i * t_max / 3
        axs[i].plot(x,u_real[round(s/dt),:])
        axs[i].plot(x,u_pred[round(s/dt),:],'--')
        axs[i].set_xlim(0,1)
        axs[i].set_ylim(-u_max,u_max)
        at = AnchoredText(r'$t=$%d'%s,
                  prop=dict(size=10), frameon=False,
                  loc='upper right',
                  )
        axs[i].add_artist(at)
        axs[i].plot(x_sensor,y_sensor,'s')
    fig.text(0.5, 0.04, r'$x$', ha='center')
    fig.text(0.04, 0.5, r'$u_r(x,t)$ and $u_p(x,t)$', va='center', rotation='vertical')
    fig.savefig('./figs/snapshots_' + file_name + '.png')
    plt.savefig('./figs/snapshots_'+file_name+'.pdf')
    figpath = './figs/snapshots_'+file_name+'.tex'
    #tikzplotlib.save(figure='gcf', filepath=figpath)
## To create animation #########################################################################
u_real = load('./data/u_real_'+file_name+'.npy')
u_pred = load('./data/u_pred_'+file_name+'.npy')
omega_b = load('./data/omega_'+file_name+'.npy')
cost_values = load('./data/cost_values_'+file_name+'.npy')
try:
    remove('./gifs/animate_'+file_name+'.gif')
except OSError:
    pass

if '2D' in equation:
    pass #animate_2D(u_real-u_pred,omega_b,'animate_'+file_name)
else:
    pass #animate(u_real,u_pred,omega_b,'animate_'+file_name)

## To create 3D plots ##########################################################################
exit()
fig = plt.figure()
ax = fig.gca(projection='3d')

error = u_real - u_pred
x, t = np.meshgrid(x, t)

surf1 = ax.plot_surface(t, x, error, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$x$')
ax.set_zlabel('Error')  

fig.savefig('./figs/error_' + file_name + '.png')
plt.savefig('./figs/error_'+file_name+'.pdf')
figpath = './figs/error_'+file_name+'.tex'
tikzplotlib.save(figure='gcf', filepath=figpath)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf1 = ax.plot_surface(t, x, u_real, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)

surf2 = ax.plot_surface(t, x, u_pred, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$x$')
ax.set_zlabel(r'$u_p$ and $u_r$')

fig.savefig('./figs/up&ur_' + file_name + '.png')
#plt.savefig('./figs/up&ur_'+file_name+'.pdf')
figpath = './figs/up&ur_'+file_name+'.tex'
#tikzplotlib.save(figure='gcf', filepath=figpath)
