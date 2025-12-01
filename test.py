from config import Particle, NBodySimulator
import numpy as np

# test config file for inital functions created

print("N-Body Simulator - Phase 1 Setup")

# test 1: single particle at rest
print("\nTest 1: Single Particle at Rest")
pos1 = [[0.5, 0.5]]
vel1 = [[0.0, 0.0]]
mass1 = [1.0]

particles1 = Particle(pos1, vel1, mass1)
print(particles1)
print(f"Kinetic energy: {particles1.kinetic_energy():.6e}")

# test 2: two particles in orbit
print("\nTest 2: Two Particles")
# simple setup - we'll refine orbital parameters later
separation = 0.2
pos2 = [[0.5 - separation/2, 0.5], 
        [0.5 + separation/2, 0.5]]
vel2 = [[0.0, 0.1], 
        [0.0, -0.1]]
mass2 = [1.0, 1.0]

particles2 = Particle(pos2, vel2, mass2)
print(particles2)
print(f"Kinetic energy: {particles2.kinetic_energy():.6e}")

# test 3: create a simulator
print("\nTest 3: Initialize Simulator")
sim = NBodySimulator(
    particles=particles1,
    box_size=1.0,
    grid_res=64,
    dt=0.01,
    boundary='periodic'
)

# test gridding
print("\nTest 4: Grid Particles")
density = sim.grid_particles()
print(f"Density grid shape: {density.shape}")
print(f"Total mass on grid: {np.sum(density) * sim.dx**2:.6f}")
print(f"Total particle mass: {np.sum(sim.particles.mass):.6f}")
if np.sum(density) * sim.dx**2 == np.sum(sim.particles.mass):
    print("they match!")
else:
    print('something is wrong :(')