import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parameters
g = 9.81
m = 1.0
L = 1.0

# linearized matrices from the paper
M = m * L**2 * np.array([
    [3, 2, 1],
    [2, 2, 1],
    [1, 1, 1]
], dtype=float)

K = m * g * L * np.array([
    [3, 0, 0],
    [0, 2, 0],
    [0, 0, 1]
], dtype=float)

Minv = np.linalg.inv(M)

def deriv(state):
    # state = [theta1, theta2, theta3, omega1, omega2, omega3]
    theta = state[:3]

    omega = state[3:]

    # Energy Loss
    damping = 0.2

    # M theta_ddot + K theta = 0
    theta_ddot = -Minv @ (K @ theta + damping * omega)

    return np.concatenate([omega, theta_ddot])

def rk4_step(state, dt):
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# initial conditions: small angles only
state = np.array([
    0.0,   # theta1
    0.0,   # theta2
    0.0,   # theta3
    0.0,    # omega1
    0.0,    # omega2
    0.0     # omega3
], dtype=float)

dt = 0.01
T = 20
steps = int(T / dt)

traj = np.zeros((steps, 6))
for i in range(steps):
    traj[i] = state
    state = rk4_step(state, dt)

theta1 = traj[:, 0]
theta2 = traj[:, 1]
theta3 = traj[:, 2]

# rod endpoint positions
x1 = L * np.sin(theta1)
y1 = -L * np.cos(theta1)

x2 = x1 + L * np.sin(theta2)
y2 = y1 - L * np.cos(theta2)

x3 = x2 + L * np.sin(theta3)
y3 = y2 - L * np.cos(theta3)

# animate
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-3.2 * L, 3.2 * L)
ax.set_ylim(-3.2 * L, 1.0 * L)
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], lw=1)

trace_x, trace_y = [], []

def update(frame):
    xs = [0, x1[frame], x2[frame], x3[frame]]
    ys = [0, y1[frame], y2[frame], y3[frame]]
    line.set_data(xs, ys)

    trace_x.append(x3[frame])
    trace_y.append(y3[frame])
    trace.set_data(trace_x, trace_y)

    return line, trace

ani = FuncAnimation(fig, update, frames=steps, interval=10, blit=True)
plt.show()