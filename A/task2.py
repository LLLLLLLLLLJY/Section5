#!/usr/bin/env python

import numpy as np
from numpy import cos, sin, sqrt, pi, dot, roll, linspace, hstack
from numpy.random import rand, seed
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from matplotlib import animation

m1, m2, l1, l2, g = 0.5, 0.7, 0.3, 0.4, 9.8

def fun(t, y):
    tht1, tht2, p1, p2 = y
    denom = m2 * (m1 + m2 - m2 * cos(tht1 - tht2) ** 2) * l1 ** 2 * l2 ** 2
    ddenom = 2 * m2 ** 2 * l1 ** 2 * l2 ** 2 * cos(tht1 - tht2) * sin(tht1 - tht2)
    num = (m2 * l2 ** 2 * p1 ** 2 / 2 + (m1 + m2) * l1 ** 2 * p2 ** 2 / 2 - 
           m2 * l1 * l2 * p1 * p2 * cos(tht1 - tht2))
    dnum = m2 * l1 * l2 * p1 * p2 * sin(tht1 - tht2)
    
    return [
        (m2 * l2 ** 2 * p1 - m2 * l1 * l2 * cos(tht1 - tht2) * p2) / denom,
        ((m1 + m2) * l1 ** 2 * p2 - m2 * l1 * l2 * cos(tht1 - tht2) * p1) / denom,
        -(dnum * denom - num * ddenom) / denom ** 2 - (m1 + m2) * g * l1 * sin(tht1),
        -(dnum * denom - num * ddenom) / denom ** 2 - m2 * g * l2 * sin(tht2),
    ]

# (c) 
sol = solve_ivp(fun, (0, 5), y0=[1, 0.1, 0, 0], vectorized=True)

plt.figure()
plt.plot(sol.y[1], sol.y[3])
plt.xlabel(r'$\theta_2$')
plt.ylabel(r'$p_2$')
plt.title("Phase Space Trajectory: $\theta_2$ vs. $p_2$")
plt.savefig("Section5/A/2_phase_space.png")
plt.close()

plt.figure()
plt.plot(sol.y[0], sol.y[1])
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.title("Phase Space Trajectory: $\theta_1$ vs. $\theta_2$")
plt.savefig("Section5/A/2_theta_trajectory.png")
plt.close()

# (c) 
def animate(sol):
    x1 = l1 * sin(sol.y[0])
    y1 = -l1 * cos(sol.y[0])
    x2 = x1 + l2 * sin(sol.y[1])
    y2 = y1 - l2 * cos(sol.y[1])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2, markersize=8)

    def update(frame):
        line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
        return line,

    anim = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=50, blit=True)
    anim.save("Section5/A/2_double_pendulum.mp4", fps=20)
    plt.close()

animate(sol)

# (d)
def rand_disk():
    r = sqrt(rand())
    theta = 2 * pi * rand()
    return r * cos(theta), r * sin(theta)

def area(points):
    vertices = ConvexHull(points).points
    x, y = vertices[:, 0], vertices[:, 1]
    return abs(dot(x, roll(y, 1)) - dot(y, roll(x, 1))) / 2

n = 20
initials = []
t_eval = linspace(0, 5, 100)

for _ in range(n):
    dtht2, dp2 = rand_disk()
    dtht2 *= 0.01
    dp2 *= 0.01
    initials.append([1, 0.1 + dtht2, 0, dp2])

solutions = [solve_ivp(fun, (0, 5), y0, vectorized=True, t_eval=t_eval) for y0 in initials]

areas = [area([[sol.y[1][i], sol.y[3][i]] for sol in solutions]) for i in range(len(t_eval))]

plt.figure()
plt.plot(t_eval, areas)
plt.xlabel(r'$t$')
plt.ylabel('Phase Space Volume')
plt.title("Evolution of Phase Space Volume")
plt.savefig("Section5/A/2_phase_space_volume.png")
plt.close()