// Simple 3-body simulation in Rust using Euler integration
// Requires Rust 1.56+ to compile with `rustc three_body_rust_simulation.rs`
// Outputs trajectories to rust_trajectories.csv for comparison

use std::fs::File;
use std::io::Write;

const G: f64 = 6.67430e-11; // gravitational constant
const MASSES: [f64; 3] = [5.0e24, 5.0e24, 5.0e24];
const TIME_STEP: f64 = 1e5; // seconds
const NUM_STEPS: usize = 300; // integration iterations

fn main() {
    // Initial positions (equilateral triangle)
    let mut positions = [
        [0.0, 0.0],
        [-0.5e11, (3f64).sqrt() * 0.5e11],
        [0.5e11, (3f64).sqrt() * 0.5e11],
    ];

    // Tangential velocities for a rotating setup
    let v0 = (G * MASSES[0] / (1e11 * (3f64).sqrt())).sqrt();
    let mut velocities = [
        [0.0, v0],
        [-v0 * (3f64).sqrt() / 2.0, -v0 / 2.0],
        [v0 * (3f64).sqrt() / 2.0, -v0 / 2.0],
    ];

    // Storage for positions at each step
    let mut trajectories = vec![[0.0; 6]; NUM_STEPS + 1];
    for i in 0..3 {
        trajectories[0][2 * i] = positions[i][0];
        trajectories[0][2 * i + 1] = positions[i][1];
    }

    // Main integration loop
    for step in 1..=NUM_STEPS {
        let mut forces = [[0.0f64; 2]; 3];
        // Compute pairwise forces
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dx = positions[j][0] - positions[i][0];
                let dy = positions[j][1] - positions[i][1];
                let dist = (dx * dx + dy * dy).sqrt();
                if dist != 0.0 {
                    let mag = G * MASSES[i] * MASSES[j] / dist.powi(2);
                    let fx = mag * dx / dist;
                    let fy = mag * dy / dist;
                    forces[i][0] += fx;
                    forces[i][1] += fy;
                    forces[j][0] -= fx;
                    forces[j][1] -= fy;
                }
            }
        }
        // Update velocities and positions
        for i in 0..3 {
            let ax = forces[i][0] / MASSES[i];
            let ay = forces[i][1] / MASSES[i];
            velocities[i][0] += ax * TIME_STEP;
            velocities[i][1] += ay * TIME_STEP;
            positions[i][0] += velocities[i][0] * TIME_STEP;
            positions[i][1] += velocities[i][1] * TIME_STEP;
            trajectories[step][2 * i] = positions[i][0];
            trajectories[step][2 * i + 1] = positions[i][1];
        }
    }

    // Write CSV output for comparison
    let mut file = File::create("rust_trajectories.csv").expect("create file");
    for row in trajectories {
        writeln!(file, "{},{},{},{},{},{}", row[0], row[1], row[2], row[3], row[4], row[5])
            .expect("write csv");
    }
}

