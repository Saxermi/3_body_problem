import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # Gravitational constant

# 2 drehen sich um 1 Körper
'''
masses = [
    1.989e30,      # Masse des zentralen Körpers (z.B. Sonne) in kg
    5.972e24,      # Masse des ersten Planeten (z.B. Erde) in kg
    6.39e23        # Masse des zweiten Planeten (z.B. Mars) in kg
]

initial_positions = np.array([
    [0, 0],               # Zentraler Körper in der Mitte
    [1.5e11, 0],          # Erster Planet, 150 Millionen km entfernt
    [-1.5e11, 0]          # Zweiter Planet auf der gegenüberliegenden Seite
], dtype='float64')

initial_velocities = np.array([
    [0, 0],               # Zentraler Körper bleibt stationär
    [0, 30000],           # Erste Planeten-Geschwindigkeit (zirkuläre Umlaufbahn)
    [0, -30000]           # Zweite Planeten-Geschwindigkeit in entgegengesetzter Richtung
], dtype='float64')

time_step = 50000     # Zeitintervall in Sekunden (14 Stunden)
num_steps = 3000       # Anzahl der Schritte für ca. 4 Jahre Simulation

'''

# Beispielwerte für ein gleichseitiges Dreieck-System
masses = [
    5.0e24,    # Masse Körper 1 in kg (z.B. ähnlich der Erde)
    5.0e24,    # Masse Körper 2 in kg
    5.0e24     # Masse Körper 3 in kg
]

# Anfangspositionen in einem gleichseitigen Dreieck mit Seitenlänge 1e11 Meter
initial_positions = np.array([
    [0, 0],                      # Körper 1
    [-0.5e11, np.sqrt(3) * 0.5e11], # Körper 2
    [-0.5e11, 0.5] # Körper 3
], dtype='float64')

# Anfangsgeschwindigkeiten tangential zu den Positionen, um Rotation zu erzeugen
v0 = np.sqrt(G * masses[0] / (1e11 * np.sqrt(3)))  # Berechnung der Umlaufgeschwindigkeit

initial_velocities = np.array([
    [0, v0],                        # Körper 1, Geschwindigkeit senkrecht zur Position
    [-v0 * np.sqrt(3)/2, -v0 / 2],  # Körper 2
    [v0 * np.sqrt(3)/2, -v0 / 2]    # Körper 3
], dtype='float64')

time_step = 10000000       # Zeitintervall in Sekunden (etwa 2.8 Stunden) --> 10000
num_steps = 5000           # Anzahl der Schritte für etwa 5 Jahre Simulation


# Initialize positions and velocities
positions = np.copy(initial_positions)
velocities = np.copy(initial_velocities)

# Arrays to store the trajectories for plotting
trajectories = [[], [], []]

def compute_gravitational_force(m1, m2, pos1, pos2):
    """ Compute gravitational force exerted on body 1 by body 2. """
    distance_vector = pos2 - pos1
    distance = np.linalg.norm(distance_vector)
    if distance == 0:  # Avoid division by zero
        return np.array([0, 0])
    force_magnitude = G * m1 * m2 / distance**2
    force_direction = distance_vector / distance
    return force_magnitude * force_direction

def update_positions_and_velocities(positions, velocities, masses, time_step):
    """ Update positions and velocities of the three bodies. """
    num_bodies = len(masses)
    forces = [np.zeros(2) for _ in range(num_bodies)]

    # Compute all pairwise forces
    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            force = compute_gravitational_force(masses[i], masses[j], positions[i], positions[j])
            forces[i] += force
            forces[j] -= force  # Newton's third law

    # Update velocities and positions based on the net force
    for i in range(num_bodies):
        acceleration = forces[i] / masses[i]
        velocities[i] += acceleration * time_step
        positions[i] += velocities[i] * time_step
        trajectories[i].append(positions[i].copy())

# Set up the figure and axis for the animation
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-2e11, 2e11)  # Set appropriate limits for visualization
ax.set_ylim(-2e11, 2e11)
colors = ['blue', 'grey', 'red']
markers = [plt.plot([], [], 'o', color=color)[0] for color in colors]

# Update function for the animation
def animate(step):
    update_positions_and_velocities(positions, velocities, masses, time_step)
    for i, marker in enumerate(markers):
        marker.set_data(positions[i][0], positions[i][1])  # Update the position of each body
        ax.plot([pos[0] for pos in trajectories[i]], [pos[1] for pos in trajectories[i]], color=colors[i], alpha=0.6)

# Create the animation
anim = FuncAnimation(fig, animate, frames=num_steps, interval=20)  # Adjust interval for speed

plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.legend(['Body 1', 'Body 2', 'Body 3'])
plt.show()
