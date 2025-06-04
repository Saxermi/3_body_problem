# Load plotting and file I/O modules
using Plots              # generate PNG output
using DelimitedFiles     # write CSV files

# Gravitational constant used in the simulation
const G = 6.67430e-11  # [m^3 kg^-1 s^-2]

# Masses of the three bodies (kg)
masses = [5.0e24, 5.0e24, 5.0e24]  # identical masses

# Initial positions in meters (forming an equilateral triangle)
initial_positions = [
    0.0      0.0;                             # body 1
    -0.5e11  sqrt(3) * 0.5e11;               # body 2
    0.5e11   sqrt(3) * 0.5e11;               # body 3
]

# Initial velocities tangential to the positions for rotation (m/s)
v0 = sqrt(G * masses[1] / (1e11 * sqrt(3)))  # base magnitude
initial_velocities = [
    0.0               v0;                    # body 1 velocity
    -v0 * sqrt(3)/2  -v0/2;                  # body 2 velocity
    v0 * sqrt(3)/2   -v0/2;                  # body 3 velocity
]

# Integration parameters
const TIME_STEP = 1e5      # seconds per step
const NUM_STEPS = 300      # number of iterations

# Current state arrays (positions and velocities)
positions = deepcopy(initial_positions)   # mutable copy of positions
velocities = deepcopy(initial_velocities) # mutable copy of velocities

# Storage for trajectories (each row: x1,y1,x2,y2,x3,y3)
trajectories = zeros(NUM_STEPS + 1, 6)
trajectories[1, :] = vec(positions)'  # store initial state

# Function to compute gravitational force between two bodies
function gravitational_force(m1, m2, pos1, pos2)
    r = pos2 .- pos1                   # displacement vector
    dist = norm(r)                     # distance between bodies
    if dist == 0.0                     # avoid division by zero
        return zeros(2)
    end
    force_mag = G * m1 * m2 / dist^2   # Newton's law of gravitation
    return force_mag .* r ./ dist      # force vector
end

# Main integration loop using a simple Euler scheme
for step in 1:NUM_STEPS
    forces = zeros(3, 2)                       # reset force accumulators
    for i in 1:3                               # loop over body pairs
        for j in i+1:3
            f = gravitational_force(masses[i], masses[j], positions[i, :], positions[j, :])
            forces[i, :] .+= f                 # apply equal and opposite forces
            forces[j, :] .-= f
        end
    end
    for i in 1:3
        acceleration = forces[i, :] ./ masses[i]
        velocities[i, :] .+= acceleration .* TIME_STEP
        positions[i, :] .+= velocities[i, :] .* TIME_STEP
    end
    trajectories[step + 1, :] = vec(positions)' # store state each step
end

# Save trajectories for comparison with Python version
writedlm("julia_trajectories.csv", trajectories, ',')  # CSV output

# Plot the paths of the three bodies
plt = plot()                               # initialize the plot
colors = [:blue, :orange, :green]          # one color per body
for i in 1:3
    x = trajectories[:, 2*(i-1)+1]         # extract x positions
    y = trajectories[:, 2*(i-1)+2]         # extract y positions
    plot!(plt, x, y, color=colors[i], label="Body $i")
end
xlabel!(plt, "x (m)")
ylabel!(plt, "y (m)")
plot!(plt, aspect_ratio = :equal)          # keep aspect ratio square
savefig(plt, "julia_trajectories.png")    # write PNG image

