import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import time

# ------------------------------------------------------------------- #
'''
m1 = 0.87
m2 = 0.8
m3 = 1.0

# Position
inital_position_1 =  [-0.5,  0.0,  0.0]
inital_position_2 =  [0.5,  0.0,  0.0]
inital_position_3 =  [0.0,   0.001, 1.0]

# Velocity
inital_velocity_1 =  [0.0, 0.347111, 0]
inital_velocity_2 =  [0.0, -0.347111, 0.0]
inital_velocity_3 =  [0.0, 0.0, -0.1]
'''

# 3-Body
# Massen
m1 = 0.800
m2 = 0.756
m3 = 1.000

# Gravitationskonstante G = 1 (implizit im Problem)

# Position
inital_position_1 = [-1.35024519775613e-01, 0.0, 0.0]  # r1(0) = (x1, 0) Blau
inital_position_2 = [1.0, 0.0, 0.0]            # stabil r2(0) = (1, 0) Rot
#inital_position_2 = [1.18, 0.0, 0.0]          # einer fliegt weg, die anderen zwei umkreisen sich.
inital_position_3 = [0.0, 0.0, 0.0]                     # r3(0) = (0, 0) Grün
#inital_position_3 = [0.08, 0.0, 0.0]           # alle fliegen in ein andere Richtung

# Geschwindigkeit
v1 = 2.51505829297841  # v1
v2 = 3.16396261593079e-01  # v2
v3 = -(m1 * v1 + m2 * v2) / m3  # Berechnung von r3'(0)

inital_velocity_1 = [0.0, v1, 0.0]  # r1'(0) = (0, v1)
inital_velocity_2 = [0.0, v2, 0.0]  # r2'(0) = (0, v2)
inital_velocity_3 = [0.0, v3, 0.0]  # r3'(0) = (0, -(m1*v1 + m2*v2)/m3)

initial_conditions = np.array([
    inital_position_1, inital_position_2, inital_position_3,
    inital_velocity_1, inital_velocity_2, inital_velocity_3
]).ravel()


# ------------------------------------------------------------------- #
# Define the system of ODEs (using the same system_odes definition)
def system_odes(S, t, m1, m2, m3):
    p1, p2, p3 = S[0:3], S[3:6], S[6:9]
    dp1_dt, dp2_dt, dp3_dt = S[9:12], S[12:15], S[15:18]

    f1, f2, f3 = dp1_dt, dp2_dt, dp3_dt

    df1_dt = m3*(p3 - p1)/np.linalg.norm(p3 - p1)**3 + m2*(p2 - p1)/np.linalg.norm(p2 - p1)**3
    df2_dt = m3*(p3 - p2)/np.linalg.norm(p3 - p2)**3 + m1*(p1 - p2)/np.linalg.norm(p1 - p2)**3
    df3_dt = m1*(p1 - p3)/np.linalg.norm(p1 - p3)**3 + m2*(p2 - p3)/np.linalg.norm(p2 - p3)**3

    return np.array([f1, f2, f3, df1_dt, df2_dt, df3_dt]).ravel()

# Initial conditions and time points
initial_conditions = np.array([
    inital_position_1, inital_position_2, inital_position_3,
    inital_velocity_1, inital_velocity_2, inital_velocity_3
]).ravel()

t_points = np.linspace(0, 15, 2001)

# Solve with odeint
solution = odeint(system_odes, initial_conditions, t_points, args=(m1, m2, m3))

# Now solution is a 2D array, with rows corresponding to different time points
# Extracting solutions for positions and velocities
p1x_sol = solution[:, 0]
p1y_sol = solution[:, 1]
p1z_sol = solution[:, 2]

p2x_sol = solution[:, 3]
p2y_sol = solution[:, 4]
p2z_sol = solution[:, 5]

p3x_sol = solution[:, 6]
p3y_sol = solution[:, 7]
p3z_sol = solution[:, 8]

# ------------------------------------------------------------------- #

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

planet1_plt, = ax.plot(p1x_sol, p1y_sol, p1z_sol, 'green', label='Planet 1', linewidth=1)
planet2_plt, = ax.plot(p2x_sol, p2y_sol, p2z_sol, 'red', label='Planet 2', linewidth=1)
planet3_plt, = ax.plot(p3x_sol, p3y_sol, p3z_sol, 'blue',label='Planet 3', linewidth=1)

planet1_dot, = ax.plot([p1x_sol[-1]], [p1y_sol[-1]], [p1z_sol[-1]], 'o', color='green', markersize=6)
planet2_dot, = ax.plot([p2x_sol[-1]], [p2y_sol[-1]], [p2z_sol[-1]], 'o', color='red', markersize=6)
planet3_dot, = ax.plot([p3x_sol[-1]], [p3y_sol[-1]], [p3z_sol[-1]], 'o', color='blue', markersize=6)


ax.set_title("The 3-Body Problem")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.grid()
plt.legend()

# ------------------------------------------------------------------- #


from matplotlib.animation import FuncAnimation

# -------  Animating the solutions ------- #

def update(frame):
    lower_lim = 0 # max(0, frame - 300)
    print(f"Progress: {(frame+1)/len(t_points):.1%} | 100.0 %", end='\r')

    x_current_1 = p1x_sol[lower_lim:frame+1]
    y_current_1 = p1y_sol[lower_lim:frame+1]
    z_current_1 = p1z_sol[lower_lim:frame+1]

    x_current_2 = p2x_sol[lower_lim:frame+1]
    y_current_2 = p2y_sol[lower_lim:frame+1]
    z_current_2 = p2z_sol[lower_lim:frame+1]

    x_current_3 = p3x_sol[lower_lim:frame+1]
    y_current_3 = p3y_sol[lower_lim:frame+1]
    z_current_3 = p3z_sol[lower_lim:frame+1]

    planet1_plt.set_data(x_current_1, y_current_1)  
    planet1_plt.set_3d_properties(z_current_1)

    planet1_dot.set_data([x_current_1[-1]], [y_current_1[-1]])
    planet1_dot.set_3d_properties([z_current_1[-1]])



    planet2_plt.set_data(x_current_2, y_current_2)  
    planet2_plt.set_3d_properties(z_current_2)

    planet2_dot.set_data([x_current_2[-1]], [y_current_2[-1]])
    planet2_dot.set_3d_properties([z_current_2[-1]])



    planet3_plt.set_data(x_current_3, y_current_3)  
    planet3_plt.set_3d_properties(z_current_3)

    planet3_dot.set_data([x_current_3[-1]], [y_current_3[-1]])
    planet3_dot.set_3d_properties([z_current_3[-1]])


    return planet1_plt, planet1_dot, planet2_plt, planet2_dot, planet3_plt, planet3_dot 

animation = FuncAnimation(fig, update, frames=range(0, len(t_points), 2), interval=10, blit=True)
plt.show()

#---------------------------------------------------------------------------------------------------

# Phasenraum-Plot für die drei Körper
fig_phase, axs = plt.subplots(3, 1, figsize=(8, 12))

# Phasenraum für Planet 1
# p1x_sol enthält die Positionen von Planet 1, die Geschwindigkeit v_x von Planet 1 ist in solution[:, 9]
axs[0].plot(p1x_sol, solution[:, 9], label="Planet 1", color='green')
axs[0].set_title("Phase-Space: Planet 1")
axs[0].set_xlabel("x")
axs[0].set_ylabel("v_x")
axs[0].grid(True)
axs[0].legend()

# Phasenraum für Planet 2
# p2x_sol enthält die Positionen von Planet 2, die Geschwindigkeit v_x von Planet 2 ist in solution[:, 12]
axs[1].plot(p2x_sol, solution[:, 12], label="Planet 2", color='red')
axs[1].set_title("Phase-Space: Planet 2")
axs[1].set_xlabel("x")
axs[1].set_ylabel("v_x")
axs[1].grid(True)
axs[1].legend()

# Phasenraum für Planet 3
# p3x_sol enthält die Positionen von Planet 3, die Geschwindigkeit v_x von Planet 3 ist in solution[:, 15]
axs[2].plot(p3x_sol, solution[:, 15], label="Planet 3", color='blue')
axs[2].set_title("Phase-Space: Planet 3")
axs[2].set_xlabel("x")
axs[2].set_ylabel("v_x")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------------------------

# Phasenporträt für Planet 1 (x, y-Raum mit Geschwindigkeitsvektoren)
fig, ax = plt.subplots(figsize=(8, 6))

# Positionsraum und Geschwindigkeiten für Planet 1
x = p1x_sol
y = p1y_sol
vx = solution[:, 9]  # Geschwindigkeit in x-Richtung
vy = solution[:, 10]  # Geschwindigkeit in y-Richtung

# Quiver-Plot für das Vektorfeld
ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=0.1, color='green', alpha=0.6)

# Trajektorie
ax.plot(x, y, label="Trajectory of Planet 1", color='darkgreen')

ax.set_title("Phase Portrait: Planet 1")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()
plt.show()

# Phasenporträt für Planet 2
fig, ax = plt.subplots(figsize=(8, 6))

# Positionsraum und Geschwindigkeiten für Planet 2
x = p2x_sol
y = p2y_sol
vx = solution[:, 12]  # Geschwindigkeit in x-Richtung
vy = solution[:, 13]  # Geschwindigkeit in y-Richtung

# Quiver-Plot für das Vektorfeld
ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=0.1, color='red', alpha=0.6)

# Trajektorie
ax.plot(x, y, label="Trajectory of Planet 2", color='darkred')

ax.set_title("Phase Portrait: Planet 2")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()
plt.show()

# Phasenporträt für Planet 3
fig, ax = plt.subplots(figsize=(8, 6))

# Positionsraum und Geschwindigkeiten für Planet 3
x = p3x_sol
y = p3y_sol
vx = solution[:, 15]  # Geschwindigkeit in x-Richtung
vy = solution[:, 16]  # Geschwindigkeit in y-Richtung

# Quiver-Plot für das Vektorfeld
ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=0.1, color='lightblue', alpha=0.6)

# Trajektorie
ax.plot(x, y, label="Trajectory of Planet 3", color='blue')

ax.set_title("Phase Portrait: Planet 3")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()
plt.show()
