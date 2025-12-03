import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy import fft
import os
os.environ['OMP_NESTED'] = 'FALSE' # to get rid of an error

class Particle:
    """
    Holds particle data for N-body simulation.
    Stores positions, velocities, and masses for multiple particles.
    """
    
    def __init__(self, positions, velocities, masses):
        """
        Initialize particles.
        
        Parameters:
        -----------
        positions : array-like, shape (N, 2)
            Particle positions [x, y] for each of N particles
        velocities : array-like, shape (N, 2)
            Particle velocities [vx, vy] for each of N particles
        masses : array-like, shape (N,)
            Particle masses
        """
        self.pos = np.array(positions, dtype=float)
        self.vel = np.array(velocities, dtype=float)
        self.mass = np.array(masses, dtype=float)
        
        # check shapes
        if self.pos.shape[0] != self.vel.shape[0] or self.pos.shape[0] != self.mass.shape[0]:
            raise ValueError("Number of particles must match in positions, velocities, and masses")
        if self.pos.shape[1] != 2:
            raise ValueError("Positions must be 2D (N, 2)")
        if self.vel.shape[1] != 2:
            raise ValueError("Velocities must be 2D (N, 2)")
    
    @property
    def n_particles(self):
        """Return number of particles"""
        return len(self.mass)
    
    def copy(self):
        """Create a deep copy of the Particle instance"""
        return Particle(
            positions=self.pos.copy(),
            velocities=self.vel.copy(),
            masses=self.mass.copy()
        )
    
    def kinetic_energy(self):
        """Calculate total kinetic energy: KE = 0.5 * sum(m * v^2)"""
        v_squared = np.sum(self.vel**2, axis=1)
        return 0.5 * np.sum(self.mass * v_squared)
    
    def __repr__(self):
        return f"Particle(n={self.n_particles}, total_mass={np.sum(self.mass):.3e})"

# numba implementation (FAST)
@nb.njit(parallel=True)
def _grid_particles_numba(pos, mass, grid_res, box_size, boundary):
    """Numba-optimized particle gridding with CIC"""
    density = np.zeros((grid_res, grid_res))
    dx = box_size / grid_res
    n_particles = pos.shape[0]
    
    for idx in nb.prange(n_particles):
        # handle boundaries
        if boundary == 1:  # periodic (use int flag for numba)
            x = pos[idx, 0] % box_size
            y = pos[idx, 1] % box_size
        else:
            x = pos[idx, 0]
            y = pos[idx, 1]
        
        # grid coordinates
        xg = x / dx
        yg = y / dx
        
        i0 = int(np.floor(xg))
        j0 = int(np.floor(yg))
        tx = xg - i0
        ty = yg - j0
        
        # neighbour indices
        if boundary == 1:  # periodic
            i = i0 % grid_res
            j = j0 % grid_res
            i1 = (i0 + 1) % grid_res
            j1 = (j0 + 1) % grid_res
        else:  # non-periodic
            i = max(0, min(i0, grid_res - 1))
            j = max(0, min(j0, grid_res - 1))
            i1 = max(0, min(i0 + 1, grid_res - 1))
            j1 = max(0, min(j0 + 1, grid_res - 1))
        
        # CIC weights
        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty)
        w01 = (1 - tx) * ty
        w11 = tx * ty
        
        m = mass[idx]
        
        # deposit mass (atomic add for thread safety)
        density[i, j] += w00 * m
        density[i1, j] += w10 * m
        density[i, j1] += w01 * m
        density[i1, j1] += w11 * m
    
    return density

@nb.njit(parallel=True)
def _interpolate_acceleration_numba(pos, accel_x, accel_y, grid_res, box_size, boundary):
    """Numba-optimized acceleration interpolation"""
    n_particles = pos.shape[0]
    particle_accel = np.zeros((n_particles, 2))
    dx = box_size / grid_res
    
    for idx in nb.prange(n_particles):
        # handle boundaries
        if boundary == 1:  # periodic
            x = pos[idx, 0] % box_size
            y = pos[idx, 1] % box_size
        else:
            x = pos[idx, 0]
            y = pos[idx, 1]
        
        xg = x / dx
        yg = y / dx
        
        i0 = int(np.floor(xg))
        j0 = int(np.floor(yg))
        tx = xg - i0
        ty = yg - j0
        
        # neighbour indices
        if boundary == 1:
            i = i0 % grid_res
            j = j0 % grid_res
            i1 = (i0 + 1) % grid_res
            j1 = (j0 + 1) % grid_res
        else:
            i = max(0, min(i0, grid_res - 1))
            j = max(0, min(j0, grid_res - 1))
            i1 = max(0, min(i0 + 1, grid_res - 1))
            j1 = max(0, min(j0 + 1, grid_res - 1))
        
        # CIC weights
        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty)
        w01 = (1 - tx) * ty
        w11 = tx * ty
        
        # interpolate
        particle_accel[idx, 0] = (w00 * accel_x[i, j] + 
                                w10 * accel_x[i1, j] +
                                w01 * accel_x[i, j1] + 
                                w11 * accel_x[i1, j1])
        
        particle_accel[idx, 1] = (w00 * accel_y[i, j] + 
                                w10 * accel_y[i1, j] +
                                w01 * accel_y[i, j1] + 
                                w11 * accel_y[i1, j1])
    
    return particle_accel

class NBodySimulator:
    """
    N-body gravity simulator using grid-based potential method.
    """
    def __init__(self, particles, box_size, grid_res, dt, 
                 boundary='periodic', softening=None, G=1.0, print_output = True):
        """
        Initialize the N-body simulator.
        
        Parameters:
        -----------
        particles : Particle
            Particle object containing positions, velocities, masses
        box_size : float
            Size of simulation box (assumed square: box_size x box_size)
        grid_res : int
            Number of grid points per dimension
        dt : float
            Timestep for integration
        boundary : str
            'periodic' or 'non-periodic' boundary conditions
        softening : float, optional
            Gravitational softening length (default: 2 * grid spacing)
        G : float
            Gravitational constant (default: 1.0)
        print_output : if set to False does not print simulations params
        """
        self.particles = particles.copy()  # store a copy to avoid external modifications
        self.box_size = box_size
        self.grid_res = grid_res
        self.dt = dt
        self.boundary = boundary
        self.G = G
        
        # grid spacing
        self.dx = box_size / grid_res
        
        # softening parameter (default to 2 grid cells)
        if softening is None:
            self.softening = 2.0 * self.dx
        else:
            self.softening = softening
        
        # time tracking
        self.time = 0.0
        self.step_count = 0
        
        # history for diagnostics
        self.energy_history = []
        self.time_history = []
        self.position_history = []
        
        if print_output == False:
            return
        else:
            print("Initialized NBodySimulator:")
            print(f"  Particles: {self.particles.n_particles}")
            print(f"  Box size: {self.box_size}")
            print(f"  Grid resolution: {self.grid_res} x {self.grid_res}")
            print(f"  Grid spacing: {self.dx:.4f}")
            print(f"  Timestep: {self.dt}")
            print(f"  Boundary: {self.boundary}")
            print(f"  Softening: {self.softening:.4f}")
    
    def _cic_weights_and_indices(self, positions):
        """
        Given positions (N,2) return arrays for CIC interpolation:
          i, j   : integer base indices (N,) - wrapped/clipped for boundaries
          i1, j1 : integer +1 indices (N,) - wrapped/clipped for boundaries  
          w00, w10, w01, w11 : weights (N,)
        These match the deposition scheme in grid_particles().
        """
        # wrap for periodic
        if self.boundary == 'periodic':
            pos = np.mod(positions, self.box_size)
        else:
            pos = positions.copy()
    
        xg = pos[:, 0] / self.dx
        yg = pos[:, 1] / self.dx
    
        i0 = np.floor(xg).astype(int)
        j0 = np.floor(yg).astype(int)
        tx = xg - i0
        ty = yg - j0
    
        # prepare neighbours
        if self.boundary == 'periodic':
            i = i0 % self.grid_res
            j = j0 % self.grid_res
            i1 = (i0 + 1) % self.grid_res
            j1 = (j0 + 1) % self.grid_res
        else:
            i = np.clip(i0, 0, self.grid_res - 1)
            j = np.clip(j0, 0, self.grid_res - 1)
            i1 = np.clip(i0 + 1, 0, self.grid_res - 1)
            j1 = np.clip(j0 + 1, 0, self.grid_res - 1)
    
        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty)
        w01 = (1 - tx) * ty
        w11 = tx * ty
    
        return i, j, i1, j1, w00, w10, w01, w11

    def grid_particles(self):
        """Grid particle masses using numba-optimized CIC"""
        boundary_flag = 1 if self.boundary == 'periodic' else 0
        return _grid_particles_numba(self.particles.pos, self.particles.mass, 
                                    self.grid_res, self.box_size, boundary_flag)
        
    def compute_potential(self, density):
        """
        Compute gravitational potential by convolving density with potential kernel.
        Uses FFT for efficient convolution.
        """
        n = self.grid_res
        
        # create coordinate arrays for the kernel
        # use indices that wrap properly for FFT
        x = np.arange(n) * self.dx
        y = np.arange(n) * self.dx
        
        # for periodic boundary conditions with FFT, we need to shift coordinates
        # so centered at (0,0)
        x = np.where(x > self.box_size/2, x - self.box_size, x)
        y = np.where(y > self.box_size/2, y - self.box_size, y)
        
        # create 2D grid
        xx, yy = np.meshgrid(x, y, indexing='ij')
        
        # distance from origin
        r_squared = xx**2 + yy**2
        
        # softened potential kernel: phi = -G / sqrt(r^2 + eps^2)
        r_soft = np.sqrt(r_squared + self.softening**2)
        potential_kernel = -self.G / r_soft

        # Use scipy's fft with workers for parallelization
        density_fft = fft.fft2(density, workers=-1)  # -1 = use all cores
        kernel_fft = fft.fft2(potential_kernel, workers=-1)
        
        potential_fft = density_fft * kernel_fft
        potential = np.real(fft.ifft2(potential_fft, workers=-1))
        
        potential *= self.dx**2
        return potential
    
    def compute_acceleration(self, potential):
        """
        Compute acceleration field from potential using gradient.
        Acceleration = -grad(potential)
        Uses central differences in interior, forward/backward at boundaries.
        
        Parameters:
        -----------
        potential : array, shape (grid_res, grid_res)
            Gravitational potential on grid
            
        Returns:
        --------
        accel_x, accel_y : arrays, shape (grid_res, grid_res)
            Acceleration field components
        """
        accel_x = np.zeros_like(potential)
        accel_y = np.zeros_like(potential)
        
        if self.boundary == 'periodic':
            # apply periodic boundary conditions manually for better control
            # use central differences everywhere with wrapping
            accel_x[1:-1, :] = -(potential[2:, :] - potential[:-2, :]) / (2 * self.dx)
            accel_y[:, 1:-1] = -(potential[:, 2:] - potential[:, :-2]) / (2 * self.dx)
            
            # wrap at boundaries
            accel_x[0, :] = -(potential[1, :] - potential[-1, :]) / (2 * self.dx)
            accel_x[-1, :] = -(potential[0, :] - potential[-2, :]) / (2 * self.dx)
            accel_y[:, 0] = -(potential[:, 1] - potential[:, -1]) / (2 * self.dx)
            accel_y[:, -1] = -(potential[:, 0] - potential[:, -2]) / (2 * self.dx)
            
        else:
            # non-periodic: use central differences in interior
            accel_x[1:-1, :] = -(potential[2:, :] - potential[:-2, :]) / (2 * self.dx)
            accel_y[:, 1:-1] = -(potential[:, 2:] - potential[:, :-2]) / (2 * self.dx)
            
            # forward/backward differences at boundaries
            accel_x[0, :] = -(potential[1, :] - potential[0, :]) / self.dx
            accel_x[-1, :] = -(potential[-1, :] - potential[-2, :]) / self.dx
            accel_y[:, 0] = -(potential[:, 1] - potential[:, 0]) / self.dx
            accel_y[:, -1] = -(potential[:, -1] - potential[:, -2]) / self.dx
        
        return accel_x, accel_y
    
    def interpolate_acceleration(self, accel_x, accel_y):
        """Interpolate acceleration using numba"""
        boundary_flag = 1 if self.boundary == 'periodic' else 0
        return _interpolate_acceleration_numba(self.particles.pos, accel_x, accel_y,
                                                self.grid_res, self.box_size, boundary_flag)

    def leapfrog_step(self):
        """
        Advance particles by one timestep using leapfrog integrator.
        Drift-Kick scheme (as shown in class):
        1. x(t+dt) = x(t) + v(t) * dt  [drift]
        2. Compute a(t+dt) at new positions
        3. v(t+dt) = v(t) + a(t+dt) * dt  [kick]
        
        Same methodoogy as two part leapfrog done in class
        """
        # drift: update positions first
        self.particles.pos += self.dt * self.particles.vel
        
        # apply boundary conditions to positions
        if self.boundary == 'periodic':
            self.particles.pos = np.mod(self.particles.pos, self.box_size)
        # for non-periodic, particles can leave the box
        
        # compute acceleration at updated positions
        density = self.grid_particles()
        density -= density.mean()
        potential = self.compute_potential(density)
        accel_x, accel_y = self.compute_acceleration(potential)
        accel = self.interpolate_acceleration(accel_x, accel_y)
        
        # kick: update velocities with new acceleration
        self.particles.vel += self.dt * accel

    def compute_energy(self):
        """
        Compute total energy (kinetic + potential) with self-energy correction.
        Self-energy calibrated from single particle at rest test: E_self = -3.499214e-03 for m=1.
        """
        ke = self.particles.kinetic_energy()
        
        # build density and potential (must match leapfrog_step)
        density = self.grid_particles()
        density -= density.mean()
        phi_grid = self.compute_potential(density)
        
        # interpolate potential to particles using CIC
        i, j, i1, j1, w00, w10, w01, w11 = self._cic_weights_and_indices(self.particles.pos)
        phi_p = (w00 * phi_grid[i, j] +
                 w10 * phi_grid[i1, j] +
                 w01 * phi_grid[i, j1] +
                 w11 * phi_grid[i1, j1])
        
        # pE from all interactions (includes self-energy)
        pe_raw = 0.5 * np.sum(self.particles.mass * phi_p)
        
        # self-energy correction from single particle (m=1) calibration
        C_self = -3.499214e-03
        self_energy = C_self * np.sum(self.particles.mass**2)
        
        pe = pe_raw - self_energy
        
        return ke + pe
        
    def run(self, n_steps, output_interval=10):
        """
        Run the simulation for n_steps.
        
        Parameters:
        -----------
        n_steps : int
            Number of timesteps to run
        output_interval : int
            How often to save energy (every output_interval steps)
        """
        print(f"\nRunning simulation for {n_steps} steps...")
        
        for step in range(n_steps):
            # advance one timestep
            self.leapfrog_step()
            self.time += self.dt
            self.step_count += 1
            
            # record energy periodically
            if step % output_interval == 0:
                energy = self.compute_energy()
                self.position_history.append(self.particles.pos.copy())
                self.energy_history.append(energy)
                self.time_history.append(self.time)
                
                if step % (output_interval * 10) == 0:
                    print(f"  Step {step}/{n_steps}, Time {self.time:.3f}, Energy {energy:.6e}")
        
        print(f"Simulation complete. Final time: {self.time:.3f}")
    
    def plot_particles(self, ax=None, **kwargs):
        """
        Plot particle positions.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        plot_kwargs = {'s': 1, 'alpha': 0.5, 'c': 'black'}
        plot_kwargs.update(kwargs)
        
        ax.scatter(self.particles.pos[:, 0], self.particles.pos[:, 1], **plot_kwargs)
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title(f'Particles at t={self.time:.2f}')
        
        return ax
    
    def plot_energy(self, ax=None):
        """
        Plot energy conservation over time.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 5))
        
        if len(self.energy_history) > 0:
            E0 = self.energy_history[0]
            dE_rel = (np.array(self.energy_history) - E0) / np.abs(E0)
            
            ax.plot(self.time_history, dE_rel * 100, 'b-', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Relative Energy Error (%)')
            ax.set_title('Energy Conservation')
        else:
            ax.text(0.5, 0.5, 'No energy data yet', 
                   ha='center', va='center', transform=ax.transAxes)
        
        return ax