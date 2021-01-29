from matplotlib import animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from parameters import *
from numpy import matmul, diag, array, meshgrid, ones, zeros
import ffmpy
from FTCS_2D import *
from nodes import *
from numpy.random import rand

def animate_2D(u_pred,omega,FileName):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(0,x_max)
    ax.set_ylim(0,y_max)
    ax.set_zlim(-2*u_max,u_max)

    x_grid, y_grid = meshgrid(x, y)
    x_sensor, y_sensor = meshgrid(x[nodes_2D(omega)[0]], y[nodes_2D(omega)[1]])
    omega_sensor = nodes_2D(omega)[2]

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$u_p(x,y,t)-u_r(x,y,t)$')

    time = ax.text(0.2,0.5,1.1*u_max,'$time=$0',None)

    def update_plot(frame_number, u_pred, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(x_grid, y_grid, u_pred[frame_number,:,:], cmap="magma")
        s = frame_number*dt
        time.set_text('$time=$%2.1f'%s)

    plot = [ax.plot_surface(x_grid, y_grid, u_pred[0,:,:], rstride=1, cstride=1)\
        , ax.scatter(x_grid, y_grid, omega_sensor, color='green')]

    path_mp4 = './mp4s/' + FileName + '.mp4'
    path_gif = './gifs/' + FileName + '.gif'

    anim = animation.FuncAnimation(fig, update_plot, len(t), fargs=(u_pred, plot), interval=0.1)
    anim.save(path_mp4, fps=30, extra_args=['-vcodec', 'libx264'])
    ff = ffmpy.FFmpeg(inputs = {path_mp4:None} , outputs = {path_gif: None})
    ff.run()