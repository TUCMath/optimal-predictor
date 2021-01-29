from matplotlib import animation
import matplotlib.pyplot as plt
from parameters import *
from nodes import *
from numpy import matmul, diag, array
import ffmpy


def animate(u_real,u_pred,omega,FileName):

    fig = plt.figure()
    y_min, y_max = [-u_max, u_max]
    ax = plt.axes(xlim=(0,x_max), ylim=(y_min,y_max))
    time=ax.annotate('$time=$0',xy=(0.5, 0.9 * u_max))
    line, = ax.plot([], [], lw=2)

    plt.xlabel('$x$')
    plt.ylabel('$u(x,t)$')

    plotcols = ["blue" , "red", "green"]
    plotlabels = ["Actual" , "Prediction", "Sensor"]
    plotlws = [2, 2, 3] 
    plotmarkers = ['.','.','s']
    lines = []
    
    for index in range(3):
        lobj = ax.plot([],[], 's', lw=plotlws[index], marker=plotmarkers[index], color=plotcols[index], label=plotlabels[index])[0]
        ax.legend()
        lines.append(lobj)


    def init():
        for line in lines:
            line.set_data([],[])
        return lines



    x_sensor = x[nodes(omega)]
    y_sensor = array([y_min]*len(x_sensor))

    def animate(i,dt):
        xlist = [x, x, x_sensor]
        ylist = [u_real[4*i,:], u_pred[4*i,:], y_sensor]
        s=4*i*dt
        time.set_text('$time=$%2.1f'%s)
        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 
        return lines

    path_mp4 = './mp4s/' + FileName + '.mp4'
    path_gif = './gifs/' + FileName + '.gif'

    anim = animation.FuncAnimation(fig,lambda i: animate(i,dt), init_func=init,frames=int(t_max/dt/4), interval=0.1, blit=False)
    anim.save(path_mp4, fps=30, extra_args=['-vcodec', 'libx264'])
    ff = ffmpy.FFmpeg(inputs = {path_mp4:None} , outputs = {path_gif: None})
    ff.run()