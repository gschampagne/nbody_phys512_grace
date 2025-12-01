# 2D n-body code for PHYS 512 final project.

## Project Overview
Built an N-body gravity simulator using a grid based potential rather than direct particle-particle force calculations as outlined in `project_guidelines.pdf`. 

## Table of Contents
- [Core Algorithm](#core-algorithm)
- [Requirements](#requirements)
- [Usage](#usage)
- [Phase 1: Setup and Classes](#phase-1)
- [Phase 2: Physics Implementation](#phase-2)
- [Phase 3: Large Simulations](#phase-3)
- [Results Summary](#results-summary)
- [Technical Notes](#technical-notes)
- [Course Info](#course-information)

## Core Algorithm
Methodology:
1. Grid the particles: convert particle positions to a density field on a grid
2. Convolve density with the (softened) potential from a single particle: get gravitational potential φ
3. Take gradient of potential: get acceleration field
4. Integrate particle motion: update positions and velociety using leapfrog solver with a fixed timestep

The project was completed in phases described below.

## Requirements
- NumPy
- Matplotlib

## Usage
The file `config.py` includes all necessary functions for the simulation.\
To run the full project pipeline:
```bash
python test.py                              # Phase 1 validation
python single_particle_and_circular_orbit.py # Phase 2 tests
python large_simulations.py                 # Phase 3 simulations
```

## Phase 1
### Setup and Classes
Using techniques learned in class, created class structure in `config.py`.

#### Class: Particle
Initializes particle positions, velocities, and masses then validates array shapes.

Functions
* n_particles: counts number of particles by counting number of masses
* copy: creates a copy of the particle instance
* kinetic energy: calculates `KE = 1/2mv^2`

Outputs the nuber of particles and thier total mass.

#### Class: NBodySImulator
Initializes particle information, box size, grid resolution, timestep, boundary and gravitational constant (we keep at 1 for simplicity). Also defines softening paramter, if not manually set, is automatically set to 2 grid cells. Starts time and step count for tracking and initializes arrays for energy and time history for diagnositics.

note: for interpolation a cubic spline was attempted but did not have good results :(

Functions Implemented:
* grid_particles: grids particle masses onto a 2D density field using the cloud in cell method (explained more in `interpolation_acceleration`). returns mass density on grid.
* plot_particles: plots particle positions
* plot_energy: plots energy conservation over time
* run: runs simulation for `n` number of timesteps and saves + prints energy in intervals specified by `output_interval`.

### Test
Successfully tested using `test_phase.py` and got output:
```
Test 1: Single Particle at Rest
Particle(n=1, total_mass=1.000e+00)
Kinetic energy: 0.000000e+00
```
```
Test 2: Two Particles
Particle(n=2, total_mass=2.000e+00)
Kinetic energy: 1.000000e-02
```
```
Test 3: Initialize Simulator
Initialized NBodySimulator:
  Particles: 1
  Box size: 1.0
  Grid resolution: 64 x 64
  Grid spacing: 0.0156
  Timestep: 0.01
  Boundary: periodic
  Softening: 0.0312
```
```
Test 4: Grid Particles
Density grid shape: (64, 64)
Total mass on grid: 1.000000
Total particle mass: 1.000000
they match!
```

## Phase 2
### Physics Implementation
Added to class NBodySimulator the following functions located in `config.py`.
#### _cic_weights_and_indices
Helper function that calculates Cloud-in-Cell interpolation weights and grid indices for particle positions. Returns the base indices (i, j), neighbor indices (i1, j1), and four bilinear weights (w00, w10, w01, w11) used in both gridding and interpolation steps.
* first order accurate (2D cubic spline would be slower)

#### compute_potential
Computed graviational potential by convolving density with softened potential from a single particle. FFT is used for the convolution sicne we have done this before in class + assignments.
```math
\phi = F^{-1}[ F[ρ] · F[φ] ]
```
wehre F represents the fourier transform.

Softened potential from a single particle is 
```math
\phi_{single} = - \frac{G}{\sqrt{r^2 + \epsilon^2}}
```
where `r` = length from the particle and `ε` = softening length. The softening is to prevent infinite force at r=0. 

By convoluting the density field with `φ_single` we get the sum of contributions from all particles which is total potential `φ`. 

#### compute_acceleration
Computes acceleration field from potential by taking gradient `-∇φ` using finite differences.

For the interior points use central difference
```math
\frac{\partial \phi}{\partial x} ≈ [\phi(i+1,j) - \phi(i-1,j)] / (2\delta x)
```
At boundary use forward and backward differences
```math
\frac{\partial \phi}{\partial x} ≈ [\phi(i+1,j) - \phi(i,j)] / (\delta x) \text{    (Forward)}
```
```math
\frac{\partial \phi}{\partial x} ≈ [\phi(i,j) - \phi(i-1,j)] / (\delta x) \text{    (Backward)}
```

#### interpolate_acceleration
Maps acceleration from grid back to particle positions using cloud in cell method to match grid_particles(). 

#### leapfrog_step
Use two step leapfrog to compute energy (like we did in class)
```python
x = x + v * dt              # Drift: update positions
dr = x[0,:] - x[1,:]        # Compute separation at NEW positions
a = dr / r**3               # Compute acceleration at NEW positions
v[0,:] = v[0,:] - a * dt    # Kick: update velocities
v[1,:] = v[1,:] + a * dt
```
Compute energy at the half-timestep position to get a more accurate energy estimate because in leapfrog, x and v are naturally offset by dt/2. 

#### compute_energy
Caluclates kinetic and potential energy
```math
\text{KE} = \frac{1}{2} mv^2
```
from our Particle class function for KE.

```math
\text{PE}_{\text{raw}} = \frac{1}{2} \sum_i m_i \phi(x_i)
```
where `m` is mass, `x` is position, `phi` is our potential we found by convolution.

but the raw potential energy includes spurious self-interaction from the grid interpolation. We subtract a calibrated self-energy term:
```math
E_{self} = C \sum_i m_i^2
```
```math
\text{PE} = \text{PE}_{\text{raw}} - E_{self}
```
where `C = -3.499214×10^3` (determined from single particle calibration).

Total energy is
```math
E = \text{KE} + \text{PE}
```

### Test
Successfully simulated a single particle at rest and two particles in a circular orbit (as outlined in Part 1 and Part 2 of the guidelines) using `single_particle_and_circular_orbit.py` and got output:
```
Test 1: Single particle at rest
Initialized NBodySimulator:
  Particles: 1
  Box size: 1.0
  Grid resolution: 64 x 64
  Grid spacing: 0.0156
  Timestep: 0.001
  Boundary: periodic
  Softening: 0.0312

Initial position: [0.5 0.5]
Initial velocity: [0. 0.]

Running simulation for 100 steps...
  Step 0/100, Time 0.001, Energy -1.600000e+01
Simulation complete. Final time: 0.100

Final position: [0.5 0.5]
Final velocity: [ 0.00000000e+00 -1.13686838e-14]
Position drift: 0.00e+00
Velocity drift: 1.14e-14
PASS: Particle remains at rest
```
```
Test 2: Binary Orbit
Finding best circular orbit velocity
Best velocity found: 1.0402
Theoretical velocity: 1.5811
Ratio: 0.6579
Separation: 0.4
Orbital velocity (each): 1.0402
Expected period: 1.2080
Initialized NBodySimulator:
  Particles: 2
  Box size: 1.0
  Grid resolution: 128 x 128
  Grid spacing: 0.0078
  Timestep: 0.001
  Boundary: periodic
  Softening: 0.1000

Initial energy: -1.133296e+01
Initial separation: 0.4000

Running simulation for 1000 steps...
  Step 0/1000, Time 0.001, Energy -1.131754e+01
  Step 500/1000, Time 0.501, Energy -1.129903e+01
Simulation complete. Final time: 1.000

Final separation: 0.3998
Separation change: 0.04%
Energy drift: 0.05%
PASS: Good energy conservation!
```
![binary orbit plot](images/binary_orbit_test.png) 
Figure 1: The final position of the particles (red and blue) after simulation are shown in the left plot above and the energy over time is displayed in the right plot.

## Phase 3
Large simulations of 300,000 particles were run with periodic and on-periodic boundaries corresponding to Part 3 in the guidelines. The particles were intially randomly scattered throughout the domain. The initial velocities are taken as 0, with the option in the code to be non-zero.

### Periodic Boundary Conditions
Sucessfully ran simulation with periodic boundary conditions with output:
```
Large N Simulation - Periodic Boundaries

Generating 300000 particles...
Total mass: 1.000000
Initial KE: 9.983594e-03
Initialized NBodySimulator:
  Particles: 300000
  Box size: 1.0
  Grid resolution: 128 x 128
  Grid spacing: 0.0078
  Timestep: 0.005
  Boundary: periodic
  Softening: 0.0200

Initial energy: 9.983601e-03

Running 500 steps (this may take a minute)...

Running simulation for 500 steps...
  Step 0/500, Time 0.005, Energy 9.983601e-03
  Step 100/500, Time 0.505, Energy 9.983601e-03
  Step 200/500, Time 1.005, Energy 9.983601e-03
  Step 300/500, Time 1.505, Energy 9.983601e-03
  Step 400/500, Time 2.005, Energy 9.983601e-03
Simulation complete. Final time: 2.500

Final energy: 9.983601e-03
Energy drift: 0.00%
PASS: Good energy conservation for large N!
```
![binary orbit plot](images/large_sim_periodic.png) 
Figure 2: Large-scale simulation with periodic boundaries (300,000 particles, 500 timesteps). Left: Final particle distribution remains uniformly scattered throughout domain. Right: Energy conservation shows 0.00% drift, demonstrating excellent stability for large N.

### Non-Periodic Boundary Conditions
Sucessfully ran simulation with periodic boundary conditions with output:
```
Large N Simulation - Non-Periodic Boundaries
Initialized NBodySimulator:
  Particles: 300000
  Box size: 1.0
  Grid resolution: 128 x 128
  Grid spacing: 0.0078
  Timestep: 0.005
  Boundary: non-periodic
  Softening: 0.0200

Initial energy: 1.002456e-02

Running 500 steps...

Running simulation for 500 steps...
  Step 0/500, Time 0.005, Energy 1.002456e-02
  Step 100/500, Time 0.505, Energy 1.002465e-02
  Step 200/500, Time 1.005, Energy 1.002550e-02
  Step 300/500, Time 1.505, Energy 1.002816e-02
  Step 400/500, Time 2.005, Energy 1.003424e-02
Simulation complete. Final time: 2.500

Final energy: 1.004590e-02
Energy drift: 0.20%
Particles escaped: 107676/300000 (35.89%)
```
![binary orbit plot](images/large_sim_nonperiodic.png) 
Figure 3: Large-scale simulation with non-periodic boundaries (300,000 particles, 500 timesteps). Left: Blue particles remain inside box boundary (dashed line), red particles escaped beyond boundaries (35.89% of total). Right: Energy drift of 0.20% reflects particle loss from system.

### Comparison: Periodic vs Non-Periodic
The comparison shows key differences between boundary conditions:
- `Periodic`: Particles remain uniformly distributed, excellent energy conservation (0.00% drift)
- `Non-Periodic`: 35.89% of particles escaped, energy drift of 0.20% due to boundary effects

![comparison plot](images/large_sim_comparison.png)
Figure 4: Direct comparison of boundary conditions. Top row: Final spatial distributions showing uniform coverage (periodic) vs particle escape (non-periodic). Bottom row: Energy evolution comparing both methods—periodic maintains perfect conservation while non-periodic shows gradual drift due to escaping particles.

## Results Summary

| Test | Status | Key Metric |
|------|--------|------------|
| Single Particle at Rest | PASS | Position drift: 0.00e+00 |
| Binary Orbit | PASS | Energy drift: 0.05% |
| Large N Periodic (300k) | PASS | Energy drift: 0.00% |
| Large N Non-Periodic (300k) | PASS | Energy drift: 0.20%, 35.89% escaped |

## Technical Notes
- `Why trial-and-error for orbital velocity?` The discrete grid and softening modify the effective gravitational potential, so the theoretical circular orbit velocity needs calibration for the specific grid setup.
- `Energy conservation`: Better in periodic boundaries because no particles/energy leave the system. Non-periodic allows escape, leading to slight energy drift.
- `Performance`: 300,000 particles for 500 steps for completes both simulations in ~20-30 minutes on M1 chip with 8GB of memory and 8 cores.

## Course Information
PHYS 512 Final Project - Fall 2024\
Written by Grace Champagne (grace.champagne@mail.mcgill.ca)
