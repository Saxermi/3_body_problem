import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# 2-Body Problem
# Massen
m1 = 2
m2 = 6.2

# Positionen
initial_position_1 = [-1.35024519775613e-01, 0.0, 0.0]  # r1(0) = (x1, 0) Blau
initial_position_2 = [1.0, 0.0, 0.0]                    # r2(0) = (1, 0) Rot

# Geschwindigkeiten
v1 = 2.51505829297841  # v1
v2 = -(m1 * v1) / m2    # Erhaltung des Gesamtdrehimpulses

initial_velocity_1 = [0.0, v1, 0.0]  # r1'(0) = (0, v1)
initial_velocity_2 = [0.0, v2, 0.0]  # r2'(0) = (0, -(m1*v1)/m2)

initial_conditions = np.array([
    initial_position_1, initial_position_2,
    initial_velocity_1, initial_velocity_2
]).ravel()

# ------------------------------------------------------------------- #
# Kraftberechnung (Beschleunigung a)
def system_odes(t, S, m1, m2):
    p1, p2 = S[0:3], S[3:6]  # Positionen
    dp1_dt, dp2_dt = S[6:9], S[9:12]  # Geschwindigkeiten

    f1, f2 = dp1_dt, dp2_dt

    df1_dt = m2 * (p2 - p1) / np.linalg.norm(p2 - p1)**3
    df2_dt = m1 * (p1 - p2) / np.linalg.norm(p1 - p2)**3

    return np.array([f1, f2, df1_dt, df2_dt]).ravel()

# ------------------------------------------------------------------- #

time_s, time_e = 0, 15
t_points = np.linspace(time_s, time_e, 2001)

solution = solve_ivp(
    fun=system_odes,
    t_span=(time_s, time_e),
    y0=initial_conditions,
    t_eval=t_points,
    args=(m1, m2)
)

# Extrahiere die Lösung
t_sol = solution.t
p1x_sol, p1y_sol, p1z_sol = solution.y[0], solution.y[1], solution.y[2]
p2x_sol, p2y_sol, p2z_sol = solution.y[3], solution.y[4], solution.y[5]

# ------------------------------------------------------------------- #
# Plot der Bahnkurven
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

planet1_plt, = ax.plot(p1x_sol, p1y_sol, p1z_sol, 'blue', label='Körper 1', linewidth=1)
planet2_plt, = ax.plot(p2x_sol, p2y_sol, p2z_sol, 'red', label='Körper 2', linewidth=1)

planet1_dot, = ax.plot([p1x_sol[-1]], [p1y_sol[-1]], [p1z_sol[-1]], 'o', color='blue', markersize=6)
planet2_dot, = ax.plot([p2x_sol[-1]], [p2y_sol[-1]], [p2z_sol[-1]], 'o', color='red', markersize=6)

ax.set_title("Das 2-Körper-Problem")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()
plt.grid()

# ------------------------------------------------------------------- #
# Animation
def update(frame):
    print(f"Progress: {(frame+1)/len(t_points):.1%} | 100.0 %", end='\r')

    x1, y1, z1 = p1x_sol[:frame+1], p1y_sol[:frame+1], p1z_sol[:frame+1]
    x2, y2, z2 = p2x_sol[:frame+1], p2y_sol[:frame+1], p2z_sol[:frame+1]

    planet1_plt.set_data(x1, y1)
    planet1_plt.set_3d_properties(z1)

    planet1_dot.set_data([x1[-1]], [y1[-1]])
    planet1_dot.set_3d_properties([z1[-1]])

    planet2_plt.set_data(x2, y2)
    planet2_plt.set_3d_properties(z2)

    planet2_dot.set_data([x2[-1]], [y2[-1]])
    planet2_dot.set_3d_properties([z2[-1]])

    return planet1_plt, planet1_dot, planet2_plt, planet2_dot

animation = FuncAnimation(fig, update, frames=range(0, len(t_points), 2), interval=10, blit=True)
plt.show()

#---------------------------------------------------------------------------------------------------

# Phasenraum-Plot für die drei Körper
fig_phase, axs = plt.subplots(2, 1, figsize=(8, 12))

# Phasenraum für Planet 1
axs[0].plot(p1x_sol, solution.y[6], label="Planet 1", color='green')
axs[0].set_title("Phase-Space: Planet 1")
axs[0].set_xlabel("x")
axs[0].set_ylabel("v_x")
axs[0].grid(True)
axs[0].legend()

# Phasenraum für Planet 2
axs[1].plot(p2x_sol, solution.y[9], label="Planet 2", color='red')
axs[1].set_title("Phase-Space: Planet 2")
axs[1].set_xlabel("x")
axs[1].set_ylabel("v_x")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------

# Phasenporträt für Planet 1 (x, y-Raum mit Geschwindigkeitsvektoren)
fig, ax = plt.subplots(figsize=(8, 6))

# Positionsraum
x = p1x_sol
y = p1y_sol
vx = solution.y[6]  # Geschwindigkeit in x-Richtung
vy = solution.y[7]  # Geschwindigkeit in y-Richtung

# Quiver-Plot für das Vektorfeld
ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=0.1, color='green', alpha=0.6)

# Trajektorie
ax.plot(x, y, label="Trajectory of Planet 1", color='darkgreen')

# Gleichgewichtspunkte markieren (bei einfachen Systemen analytisch bestimmbar)
# Hier exemplarisch: Keine bekannten Gleichgewichtspunkte, da dynamisch
# ax.plot(eq_x, eq_y, 'ro', label="Equilibrium Points")

ax.set_title("Phase Portrait: Planet 1")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()
plt.show()

# Phasenporträt für Planet 2 (x, y-Raum mit Geschwindigkeitsvektoren)
fig, ax = plt.subplots(figsize=(8, 6))

# Positionsraum
x = p2x_sol
y = p2y_sol
vx = solution.y[9]  # Geschwindigkeit in x-Richtung
vy = solution.y[10]  # Geschwindigkeit in y-Richtung

# Quiver-Plot für das Vektorfeld
ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=0.1, color='red', alpha=0.6)

# Trajektorie
ax.plot(x, y, label="Trajectory of Planet 2", color='darkred')

# Gleichgewichtspunkte markieren (bei einfachen Systemen analytisch bestimmbar)
# Hier exemplarisch: Keine bekannten Gleichgewichtspunkte, da dynamisch
# ax.plot(eq_x, eq_y, 'ro', label="Equilibrium Points")

ax.set_title("Phase Portrait: Planet 2")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()
plt.show()