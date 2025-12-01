from config import Particle, NBodySimulator
import numpy as np
import matplotlib.pyplot as plt

# big simulation!
print("N-Body Simulator - Phase 3: Large Simulations")

print("\nLarge N Simulation - Periodic Boundaries")

# generate random particle distribution
n_particles = 300000
print(f"\nGenerating {n_particles} particles...")

# random positions uniformly distributed in box
pos_random = np.random.uniform(0, 1.0, size=(n_particles, 2))

# initial velocities - two options:
# option 1: zero velocities
#vel_random = np.zeros((n_particles, 2))

# option 2: random velocities 
vel_random = np.random.normal(0, 0.1, size=(n_particles, 2))

# equal mass particles
mass_random = np.ones(n_particles) / n_particles  # total mass = 1

particles_large = Particle(pos_random, vel_random, mass_random)
print(f"Total mass: {np.sum(particles_large.mass):.6f}")
print(f"Initial KE: {particles_large.kinetic_energy():.6e}")

# create simulator with periodic boundaries
sim_periodic = NBodySimulator(
    particles=particles_large,
    box_size=1.0,
    grid_res=128,
    dt=0.005,
    boundary='periodic',
    softening=0.02
)

print(f"\nInitial energy: {sim_periodic.compute_energy():.6e}")

# run simulation
n_steps_phase3 = 500
print(f"\nRunning {n_steps_phase3} steps (this may take a minute)...")
sim_periodic.run(n_steps=n_steps_phase3, output_interval=10)

print(f"\nFinal energy: {sim_periodic.compute_energy():.6e}")

# analyze energy conservation
if len(sim_periodic.energy_history) > 1:
    E0 = sim_periodic.energy_history[0]
    E_final = sim_periodic.energy_history[-1]
    dE_percent = abs(E_final - E0) / abs(E0) * 100
    print(f"Energy drift: {dE_percent:.2f}%")
    
    if dE_percent < 5.0:
        print("PASS: Good energy conservation for large N!")
    else:
        print("WARNING: Energy drift is significant")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

sim_periodic.plot_particles(ax=axes[0], s=0.1, alpha=0.5)
axes[0].set_title(f'Periodic Boundaries - Final State (t={sim_periodic.time:.2f})')

sim_periodic.plot_energy(ax=axes[1])

plt.tight_layout()
plt.savefig('images/large_sim_periodic.png', dpi=150, bbox_inches='tight')
print("\nSaved plot to: large_sim_periodic.png")

# non-periodic boundaries 
print("\nLarge N Simulation - Non-Periodic Boundaries")

# Use these settings for non-periodic only
pos_nonperiodic = np.random.uniform(size=(n_particles, 2))
vel_nonperiodic = np.random.normal(0, 0.1, size=(n_particles, 2))  # small velocities
# use same initial conditions
particles_large_np = Particle(pos_nonperiodic, vel_nonperiodic, mass_random.copy())

# create simulator with non-periodic boundaries
sim_nonperiodic = NBodySimulator(
    particles=particles_large_np,
    box_size=1.0,
    grid_res=128,
    dt=0.005,
    boundary='non-periodic',
    softening=0.02
)

print(f"\nInitial energy: {sim_nonperiodic.compute_energy():.6e}")

# run simulation
print(f"\nRunning {n_steps_phase3} steps...")
sim_nonperiodic.run(n_steps=n_steps_phase3, output_interval=10)

print(f"\nFinal energy: {sim_nonperiodic.compute_energy():.6e}")

# analyze energy conservation
if len(sim_nonperiodic.energy_history) > 1:
    E0 = sim_nonperiodic.energy_history[0]
    E_final = sim_nonperiodic.energy_history[-1]
    dE_percent = abs(E_final - E0) / abs(E0) * 100
    print(f"Energy drift: {dE_percent:.2f}%")

# count particles that escaped
escaped = np.sum((sim_nonperiodic.particles.pos[:, 0] < 0) | 
                 (sim_nonperiodic.particles.pos[:, 0] > sim_nonperiodic.box_size) |
                 (sim_nonperiodic.particles.pos[:, 1] < 0) | 
                 (sim_nonperiodic.particles.pos[:, 1] > sim_nonperiodic.box_size))
print(f"Particles escaped: {escaped}/{n_particles} ({100*escaped/n_particles:.2f}%)")

# visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# plot with extended limits to show escaped particles
ax = axes[0]
in_box = ((sim_nonperiodic.particles.pos[:, 0] >= 0) & 
          (sim_nonperiodic.particles.pos[:, 0] <= sim_nonperiodic.box_size) &
          (sim_nonperiodic.particles.pos[:, 1] >= 0) & 
          (sim_nonperiodic.particles.pos[:, 1] <= sim_nonperiodic.box_size))

ax.scatter(sim_nonperiodic.particles.pos[in_box, 0], 
           sim_nonperiodic.particles.pos[in_box, 1], 
           s=0.1, alpha=0.5, c='blue')
ax.scatter(sim_nonperiodic.particles.pos[~in_box, 0], 
           sim_nonperiodic.particles.pos[~in_box, 1], 
           s=0.5, alpha=0.8, c='red')

# draw box boundary
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', linewidth=2, label='Box boundary')

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.set_title(f'Non-Periodic Boundaries - Final State (t={sim_nonperiodic.time:.2f})')
ax.legend()

sim_nonperiodic.plot_energy(ax=axes[1])

plt.tight_layout()
plt.savefig('images/large_sim_nonperiodic.png', dpi=150, bbox_inches='tight')
print("\nSaved plot to: large_sim_nonperiodic.png")

# comparison
print("\nComparison: Periodic vs Non-Periodic")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# periodic final state
sim_periodic.plot_particles(ax=axes[0, 0], s=0.1, alpha=0.5)
axes[0, 0].set_title('Periodic - Final State')

# non-periodic final state
axes[0, 1].scatter(sim_nonperiodic.particles.pos[in_box, 0], 
                   sim_nonperiodic.particles.pos[in_box, 1], 
                   s=0.1, alpha=0.5, c='blue')
axes[0, 1].plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', linewidth=2)
axes[0, 1].set_xlim(0, 1)
axes[0, 1].set_ylim(0, 1)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].set_aspect('equal')
axes[0, 1].set_title('Non-Periodic - Final State')

# energy comparison
axes[1, 0].plot(sim_periodic.time_history, 
                100*(np.array(sim_periodic.energy_history) - sim_periodic.energy_history[0])/abs(sim_periodic.energy_history[0]),
                'b-', linewidth=2, label='Periodic')
axes[1, 0].plot(sim_nonperiodic.time_history, 
                100*(np.array(sim_nonperiodic.energy_history) - sim_nonperiodic.energy_history[0])/abs(sim_nonperiodic.energy_history[0]),
                'r-', linewidth=2, label='Non-Periodic')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Relative Energy Error (%)')
axes[1, 0].set_title('Energy Conservation Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# absolute energy
axes[1, 1].plot(sim_periodic.time_history, sim_periodic.energy_history,
                'b-', linewidth=2, label='Periodic')
axes[1, 1].plot(sim_nonperiodic.time_history, sim_nonperiodic.energy_history,
                'r-', linewidth=2, label='Non-Periodic')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Total Energy')
axes[1, 1].set_title('Absolute Energy Evolution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/large_sim_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved comparison plot to: large_sim_comparison.png")