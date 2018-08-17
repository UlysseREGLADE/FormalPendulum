import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pendulum import *
from simplex_opt import *


N = 2 # , The number of consecutives penduliums
T = 10 #s, The duration of the simulation
FPS = 25 # , The number of fram per seconds
SAVE_ANIM = False # , Do we export an .mp4 animation of the dynamic

Steps = FPS*T
t = np.linspace(0, T, Steps+1)
dt = T/Steps

#Here we solve the dynamic of th pendulium
pend = pendulum(lengths=[0.25 for i in range(N)],
                         masses=[0.5 for i in range(N+1)])

y_ref = [0, 0]
for i in range(pend.n):
    y_ref.append(np.pi/4)
    y_ref.append(0)
y_ref = np.array(y_ref)

def cost_function(u):
    pend.control = np.array([u[0], 0, u[1], 0, u[2], 0])
    ret = 0
    for i in range(1):
        #y_rnd = (np.random.random(pend.n)-0.5)
        y0 = np.array(y_ref)
        y0 += np.array([0, 0, 0.5, 0, 0.3, 0])
        sol = integrate.odeint(pend.f_from_lambda, y0, t)
        ret += np.sum(np.power(sol-y_ref, 2))
    return ret

#u = optimize(cost_function, 3, 100, rnd0 = np.array([1, 100, 100]))


#======================================================#
#
# For the 1-pendulum: [ 20776.1, 26064.9, -2098395.9, -114632.2]
#
#======================================================#

pend.control = np.zeros((2*pend.n+2))
y_rnd = (np.random.random(pend.n)-0.5)*np.pi
y0 = np.array(y_ref)
y0[pend.even_indexs[1:]] += y_rnd
#y0=y_ref
#print(y0)
sol = integrate.odeint(pend.f_from_lambda, y0, t)
#print(np.sum(np.power(sol-y_ref, 2)))

#A cool animation
pos = np.zeros((pend.n+1, 2, len(t)))
pos[0,0,:] = sol[:, 0]
for i in range(pend.n):
    pos[i+1,0,:] = pos[i,0,:] + pend.lengths[i]*np.sin(sol[:, 2*(i+1)])
    pos[i+1,1,:] = pos[i,1,:] - pend.lengths[i]*np.cos(sol[:, 2*(i+1)])

fig = plt.figure(figsize=(6, 6))
l = np.sum(np.array(pend.lengths))
ax = fig.add_subplot(111, autoscale_on=False,
                     xlim=(-l-0.25, l+0.25), ylim=(-l-0.25, l+0.25))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = pos[:, 0, i]
    thisy = pos[:, 1, i]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)),
                              interval=FPS, blit=True, init_func=init)

if(SAVE_ANIM):
    ani.save('3_pendulum.mp4', fps=FPS)

plt.show()
