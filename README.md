# 3_body_problem
## File origin
- n_body_problem_3D --> https://github.com/Younes-Toumi/Youtube-Channel/tree/main/Simulation%20with%20Python/3%20Body%20Problem
- n_body_problem_3D_gif_version --> https://github.com/Younes-Toumi/Modeling-and-Simulating-Complex-Chaotic-Systems/tree/main/N-Body-Problem

## Julia Simulation
A simple Julia implementation of the 3-body problem is provided in `three_body_julia_simulation.jl`. The script uses `Plots` and `DelimitedFiles` from the Julia standard ecosystem.

### Requirements
- Julia 1.9 or later
- `Plots` package (`] add Plots` from the Julia REPL)

### Running
Execute the simulation with

```bash
julia three_body_julia_simulation.jl
```

The script writes `julia_trajectories.csv` and creates `julia_trajectories.png` showing the three trajectories.

To compare with the Python version run

```bash
python simulate_python.py
```

Running the Python script creates `python_trajectories.csv` and a PNG
`python_trajectories.png` of the Python paths. If Rust or Julia data is
present, additional plots will be produced:

- `python_rust_comparison.png` – Python vs. Rust
- `comparison.png` – overlay of Python, Rust and Julia (if available)

These images appear in the repository directory.

### Installation notes
On some minimal environments `apt-get install julia` returns `Package julia has no installation candidate`.
To install Julia you can add the official repository:

```bash
sudo apt-get update
sudo apt-get install wget ca-certificates
wget -qO- https://julialang-s3.julialang.org/bin/linux/debian/archive.key | sudo tee /etc/apt/trusted.gpg.d/julia.asc
sudo sh -c "echo 'deb https://julialang-s3.julialang.org/bin/linux/ubuntu $(lsb_release -cs) main' > /etc/apt/sources.list.d/julia.list"
sudo apt-get update
sudo apt-get install julia
```

Or manually download the tarball if network access is allowed:

```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.2-linux-x86_64.tar.gz
 tar xzf julia-1.10.2-linux-x86_64.tar.gz
 sudo mv julia-1.10.2 /opt/julia
 sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia
```

Make sure Python dependencies are installed before running `simulate_python.py`:

```bash
pip install numpy matplotlib
```

## Rust Simulation
A minimal Rust version is available in `three_body_rust_simulation.rs`.

### Requirements
- Rust toolchain (rustc and cargo)

### Running
Compile and run the program with

```bash
rustc three_body_rust_simulation.rs
./three_body_rust_simulation
```

The program writes `rust_trajectories.csv`. Run `python simulate_python.py`
afterwards to create `python_rust_comparison.png` and the full
`comparison.png`.
