import numpy as np
import matplotlib.pyplot as plt
from config import Particle, NBodySimulator

# define functions to set inital positions of particles
def make_clustered_ic(N, box_size, mass_total=1.0, overdense_radius=0.15, center=(0.5, 0.5),
                      vel_dispersion=0.02, background_fraction=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # split particles between cluster and background
    N_cluster = int((1.0 - background_fraction) * N)
    N_bg = N - N_cluster

    # cluster: isotropic Gaussian around the chosen center
    cx, cy = center
    cluster_pos = rng.normal(loc=[cx, cy], scale=overdense_radius / 2.0, size=(N_cluster, 2))
    cluster_pos = np.mod(cluster_pos, box_size)

    # background
    bg_pos = rng.random((N_bg, 2)) * box_size
    positions = np.vstack([cluster_pos, bg_pos])
    
    velocities = vel_dispersion * rng.normal(size=(N, 2)) # cold velocities
    masses = np.ones(N, dtype=float) # equal mass
    masses *= mass_total / np.sum(masses)

    return positions, velocities, masses


def make_sinusoidal_ic(N, box_size, mass_total=1.0, mode_k=2, vel_dispersion=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    A = 0.8  # amplitude of the perturbation

    # sample x from a perturbed distribution using rejection sampling
    N_target = N
    xs = []
    while len(xs) < N_target:
        x_trial = rng.random() * box_size
        u = rng.random()
        density_factor = 1.0 + A * np.cos(2.0 * np.pi * mode_k * x_trial / box_size)
        if u < density_factor / (1.0 + A):
            xs.append(x_trial)
    xs = np.array(xs)

    # y uniform
    ys = rng.random(N_target) * box_size
    positions = np.column_stack([xs, ys])

    velocities = vel_dispersion * rng.normal(size=(N_target, 2)) # small random velocities
    masses = np.ones(N_target, dtype=float) # equal masses
    masses *= mass_total / np.sum(masses)

    return positions, velocities, masses

# running cluster collapse
N = 300000
box_size = 1.0
grid_res = 128
dt = 0.001
n_steps = 1000

rng = np.random.default_rng(1234)
pos, vel, m = make_clustered_ic(
    N=N,
    box_size=box_size,
    overdense_radius=0.12,
    center=(0.5, 0.5),
    vel_dispersion=0.01,
    background_fraction=0.5,
    rng=rng,
)
particles = Particle(positions=pos, velocities=vel, masses=m)

sim = NBodySimulator(
    particles=particles,
    box_size=box_size,
    grid_res=grid_res,
    dt=dt,
    boundary='periodic',
    softening=0.02,
)
sim.run(n_steps=n_steps, output_interval=10)

# plot final position + energy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

sim.plot_particles(ax=ax1, s=0.1, alpha=0.4, c='black')
ax1.set_title("Cluster Collapse – Final State")
sim.plot_energy(ax=ax2)
ax2.set_title("Energy Conservation (Cluster IC)")
plt.tight_layout()
#plt.savefig("images/large_sim_cluster.png")
plt.show()


# running sinusoidal
N = 300000
box_size = 1.0
grid_res = 128
dt = 0.001
n_steps = 1000

rng = np.random.default_rng(5678)
pos, vel, m = make_sinusoidal_ic(
    N=N,
    box_size=box_size,
    mode_k=2,
    vel_dispersion=0.005,
    rng=rng,
)
particles = Particle(positions=pos, velocities=vel, masses=m)

sim = NBodySimulator(
    particles=particles,
    box_size=box_size,
    grid_res=grid_res,
    dt=dt,
    boundary='periodic',
    softening=0.02,
)
sim.run(n_steps=n_steps, output_interval=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# plot final position + energy
sim.plot_particles(ax=ax1, s=0.1, alpha=0.4, c='black')
ax1.set_title("Sinusoidal Mode – Final State")
sim.plot_energy(ax=ax2)
ax2.set_title("Energy Conservation (Sinusoidal IC)")
plt.tight_layout()
#plt.savefig("images/large_sim_sinusoidal.png")
plt.show()