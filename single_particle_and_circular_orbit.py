from config import Particle, NBodySimulator
import numpy as np
import matplotlib.pyplot as plt

# test config file for first simulation (particle at rest and binary orbit)
print("N-Body Simulator - Phase 2: Physics Implementation")

# test 1: single particle
print("\nTest 1: Single particle at rest")
pos1, vel1, mass1 = [[0.5, 0.5]],  [[0.0, 0.0]], [1.0]
particles1 = Particle(pos1, vel1, mass1)

sim1 = NBodySimulator(
    particles=particles1,
    box_size=1.0,
    grid_res=128,
    dt=0.001,  # small timestep for accuracy
    boundary='periodic',
    softening = 0.02
)

print(f"\nInitial position: {sim1.particles.pos[0]}")
print(f"Initial velocity: {sim1.particles.vel[0]}")

# run simulation for a short time
sim1.run(n_steps=100, output_interval=10)
print(f"\nFinal position: {sim1.particles.pos[0]}")
print(f"Final velocity: {sim1.particles.vel[0]}")
print(f"Position drift: {np.linalg.norm(sim1.particles.pos[0] - np.array([0.5, 0.5])):.2e}")
print(f"Velocity drift: {np.linalg.norm(sim1.particles.vel[0]):.2e}")

if np.linalg.norm(sim1.particles.vel[0]) < 1e-6:
    print("PASS: Particle remains at rest")
else:
    print("FAIL: Particle is moving :(")

# test 2: binary orbit
print("\nTest 2: Binary Orbit")
# had issues with theoretical velocity so created a trial and error to find best one
def find_circular_orbit_velocity(separation, mass, box_size, grid_res, softening, v_guess, n_test_steps=200, dt=0.001):
    print("Finding best circular orbit velocity")
    best_v = v_guess
    best_drift = float('inf')
    # try velocities around the guess
    v_trials = v_guess * np.linspace(0.5, 1.5, 20)
    for v_trial in v_trials:
        #test orbit
        pos_test = [[0.5 - separation/2, 0.5], 
                    [0.5 + separation/2, 0.5]]
        vel_test = [[0.0, v_trial], 
                    [0.0, -v_trial]]
        mass_test = [mass, mass]
        particles_test = Particle(pos_test, vel_test, mass_test)
        
        sim_test = NBodySimulator(
            particles=particles_test,
            box_size=box_size,
            grid_res=grid_res,
            dt=dt,
            boundary='periodic',
            softening=softening,
            print_output = False
        )
        # run for short time
        for _ in range(n_test_steps):
            sim_test.leapfrog_step()
            sim_test.time += sim_test.dt
            sim_test.step_count += 1
        
        # check final separation
        final_sep = np.linalg.norm(sim_test.particles.pos[0] - sim_test.particles.pos[1])
        drift = abs(final_sep - separation)
        if drift < best_drift:
            best_drift = drift
            best_v = v_trial
    print(f"Best velocity found: {best_v:.4f}")
    print(f"Theoretical velocity: {v_guess:.4f}")
    print(f"Ratio: {best_v/v_guess:.4f}")
    return best_v

# set up circular orbit
separation = 0.4
m1 = m2 = 1.0
M_total = m1 + m2
G = 1.0

# theoretical orbital velocity for comparison
v_theory = np.sqrt(G * M_total / (2 * separation))
v_orbit = find_circular_orbit_velocity(
    separation=separation,
    mass=m1,
    box_size=1.0,
    grid_res=128,
    softening=0.02,
    v_guess=v_theory,
    n_test_steps=200,
    dt=0.001
)
print(f"Separation: {separation}")
print(f"Orbital velocity (each): {v_orbit:.4f}")
print(f"Expected period: {2 * np.pi * separation / (2 * v_orbit):.4f}")

pos2 = [[0.5 - separation/2, 0.5], [0.5 + separation/2, 0.5]]
vel2 = [[0.0, v_orbit], [0.0, -v_orbit]]
mass2 = [m1, m2]
particles2 = Particle(pos2, vel2, mass2)

sim2 = NBodySimulator(
    particles=particles2,
    box_size=1.0,
    grid_res=128,
    dt=0.001,
    boundary='periodic',
    softening=0.02
)
print(f"\nInitial energy: {sim2.compute_energy():.6e}")
print(f"Initial separation: {np.linalg.norm(sim2.particles.pos[0] - sim2.particles.pos[1]):.4f}")

# run for a few orbits
sim2.run(n_steps=1000, output_interval=50)

final_separation = np.linalg.norm(sim2.particles.pos[0] - sim2.particles.pos[1])
print(f"\nFinal separation: {final_separation:.4f}")
print(f"Separation change: {abs(final_separation - separation)/separation * 100:.2f}%")

# check energy conservation
if len(sim2.energy_history) > 1:
    E0 = sim2.energy_history[0]
    E_final = sim2.energy_history[-1]
    dE_percent = abs(E_final - E0) / abs(E0) * 100
    print(f"Energy drift: {dE_percent:.2f}%")
    
    if dE_percent < 5.0:
        print("PASS: Good energy conservation!")
    else:
        print("SAD: Energy drift is significant")

# create visualization
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
# plot final state
sim2.plot_particles(ax=axes[0], s=100, c=['red', 'blue'])
axes[0].plot([pos2[0][0], pos2[1][0]], [pos2[0][1], pos2[1][1]], 'k--', alpha=0.3, label='Initial')
axes[0].legend()
axes[0].set_title('Binary Orbit - Final State')
sim2.plot_energy(ax=axes[1])
plt.tight_layout()
#plt.savefig('images/binary_orbit_test.png')
plt.show()