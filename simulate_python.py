import numpy as np
import matplotlib.pyplot as plt
import os

# Gravitational constant
G = 6.67430e-11

# All bodies have the same mass
masses = np.array([5.0e24, 5.0e24, 5.0e24])

# Initial positions forming an equilateral triangle (meters)
initial_positions = np.array([
    [0.0, 0.0],
    [-0.5e11, np.sqrt(3) * 0.5e11],
    [0.5e11, np.sqrt(3) * 0.5e11],
], dtype=float)

# Tangential velocities for a rotating configuration (m/s)
v0 = np.sqrt(G * masses[0] / (1e11 * np.sqrt(3)))
initial_velocities = np.array([
    [0.0, v0],
    [-v0 * np.sqrt(3)/2, -v0 / 2],
    [v0 * np.sqrt(3)/2, -v0 / 2],
], dtype=float)

# Integration parameters
TIME_STEP = 1e5  # seconds
NUM_STEPS = 300  # number of iterations

# Prepare arrays for positions and velocities
positions = initial_positions.copy()
velocities = initial_velocities.copy()
trajectories = np.zeros((NUM_STEPS+1, 6))
trajectories[0] = positions.flatten()

# Compute gravitational force between two bodies

def gravitational_force(m1, m2, pos1, pos2):
    r = pos2 - pos1
    dist = np.linalg.norm(r)
    if dist == 0:
        return np.zeros(2)
    force_mag = G * m1 * m2 / dist**2
    return force_mag * r / dist

# Perform the integration using a simple Euler scheme
for step in range(1, NUM_STEPS+1):
    forces = np.zeros((3, 2))
    for i in range(3):
        for j in range(i+1, 3):
            f = gravitational_force(masses[i], masses[j], positions[i], positions[j])
            forces[i] += f
            forces[j] -= f
    for i in range(3):
        accel = forces[i] / masses[i]
        velocities[i] += accel * TIME_STEP
        positions[i] += velocities[i] * TIME_STEP
    trajectories[step] = positions.flatten()

# Save results for later comparison
np.savetxt('python_trajectories.csv', trajectories, delimiter=',')

# Load Julia results if present and compute difference
julia_data = None
if os.path.exists('julia_trajectories.csv'):
    julia_data = np.loadtxt('julia_trajectories.csv', delimiter=',')

rust_data = None
if os.path.exists('rust_trajectories.csv'):
    rust_data = np.loadtxt('rust_trajectories.csv', delimiter=',')

labels = ['Body 1', 'Body 2', 'Body 3']
colors = ['tab:blue', 'tab:orange', 'tab:green']

# Plot Python trajectories only
plt.figure()
for i in range(3):
    x = trajectories[:, 2*i]
    y = trajectories[:, 2*i+1]
    plt.plot(x, y, color=colors[i], label=f'Python {labels[i]}')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig('python_trajectories.png')

# Plot comparison of available implementations
plt.figure()
for i in range(3):
    x = trajectories[:, 2*i]
    y = trajectories[:, 2*i+1]
    plt.plot(x, y, color=colors[i], label=f'Python {labels[i]}')
    if julia_data is not None:
        plt.plot(julia_data[:, 2*i], julia_data[:, 2*i+1], '--', color=colors[i], label=f'Julia {labels[i]}')
    if rust_data is not None:
        plt.plot(rust_data[:, 2*i], rust_data[:, 2*i+1], ':', color=colors[i], label=f'Rust {labels[i]}')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig('comparison.png')

# Plot Python vs Rust only if rust data is present
if rust_data is not None:
    plt.figure()
    for i in range(3):
        plt.plot(trajectories[:, 2*i], trajectories[:, 2*i+1], color=colors[i], label=f'Python {labels[i]}')
        plt.plot(rust_data[:, 2*i], rust_data[:, 2*i+1], ':', color=colors[i], label=f'Rust {labels[i]}')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('python_rust_comparison.png')

# Print final position difference if Julia data available
if julia_data is not None:
    diff = np.linalg.norm(trajectories[-1] - julia_data[-1])
    print(f'Final position vector difference Python vs Julia: {diff:.3e} m')
else:
    print('Julia results not found. Run the Julia script before comparison.')

if rust_data is not None:
    diff_rust = np.linalg.norm(trajectories[-1] - rust_data[-1])
    print(f'Final position vector difference Python vs Rust: {diff_rust:.3e} m')
else:
    print('Rust results not found. Run the Rust program before comparison.')
